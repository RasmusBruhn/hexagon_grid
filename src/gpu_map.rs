use super::{
    INV_SQRT_3, SQRT_3,
    types::{Point, Size, Index, View, Transform2D},
    color::ColorMap,
    render::RenderState,
    map,
};
use std::{mem, f64};
use wgpu::util::DeviceExt;
use thiserror::Error;

/// A GPU representation of a map, it does not save a map internally, this must be supplied whenever id's are updated
pub struct GPUMap {
    /// All data related to the view and screen transform
    view: GPUMapView,
    /// All the buffers for producing hexagons
    hex_buffers: GPUMapHexBuffers,
    /// All of the location and id buffers for the map
    data_buffers: GPUMapDataBuffers,
    /// All of the uniforms
    uniforms: GPUMapUniforms,
    /// All of the pipelines
    pipelines: GPUMapPipelines,
}

/// Describes what part of the map is currently loaded and how to render the map (transforms and extra buffer space)
struct GPUMapView {
    /// The size of a chunk (radius)
    size: usize,
    /// The currently loaded view, when the visible region goes outside of this then a reload is needed
    view: View,
    /// The amount of the map to load relative to the shown area, must be at least 1
    map_buffer_size: f64,
    /// The transform from world coordinates to the screen
    transform: Transform2D,
    /// The index of the center chunk
    index_center: Index,
    /// Values related to the number of chunks loaded in the x and y direction
    index_size: Index,
}

/// Holds GPU buffers for the vertex data to draw a hexagon
struct GPUMapHexBuffers {
    /// The hex vertex buffer
    vertices: wgpu::Buffer,
    /// The buffer holding all of the bulk index data for a hexagon
    indices_bulk: wgpu::Buffer,
    /// The buffer holding all of the edge index data for a hexagon
    indices_edge: wgpu::Buffer,
}

/// Holds all of the data buffers with information about the map
struct GPUMapDataBuffers {
    /// All the chunk data
    chunk: GPUMapDataBufferList,
    /// All the edge data
    edges: [GPUMapDataBufferList; 3],
    /// All the vertex data
    vertices: [GPUMapDataBufferList; 2],
}

/// Holds all the data buffers with information about all the chunks, edges (0/1/2) or vertices (0/1)
struct GPUMapDataBufferList {
    /// The locations buffer
    locations: wgpu::Buffer,
    /// The buffers for id and origin
    buffers: Vec<GPUMapDataBuffer>,
}

/// Holds the data for rendering one part of the map, a single chunk, edge or vertex
struct GPUMapDataBuffer {
    /// The number of hexagons
    size: usize,
    /// The origin of the hexagons (center coordinate is: location + origin)
    origin_data: Point,
    /// The id of each of the hexagons
    ids: wgpu::Buffer,
    /// The buffer to write origin to the gpu
    origin: wgpu::Buffer,
    /// The bind group for the origin
    bind_group_origin: wgpu::BindGroup,
}

/// Holds all of the global uniforms for the shader and the bind group for them
struct GPUMapUniforms {
    /// The transformation matrix
    center_transform: wgpu::Buffer,
    /// The draw mode buffer
    draw_mode: wgpu::Buffer,
    /// The color map buffer
    _color_map: wgpu::Buffer,
    /// The bind group for the center transform and the draw mode
    bind_group: wgpu::BindGroup,
}

/// Holds both of the render pipelines for filling and outline rendering
struct GPUMapPipelines {
    /// The render pipeline for filling
    fill: wgpu::RenderPipeline,
    /// The render pipeline for the outline
    outline: wgpu::RenderPipeline,
}

impl GPUMap {
    /// Creates a new GPUMap
    /// 
    /// # Parameters
    /// 
    /// map_buffer_size: The ratio of the loaded size over the rendered size, used to load more than needed to avoid constant reloading
    /// 
    /// transform: The initial transform from world coordinates to screen coordinates
    /// 
    /// color_map: The color map to render the tiles with
    /// 
    /// map: The map to render
    /// 
    /// shader: The shader program to use
    /// 
    /// render_state: The render state to use for rendering
    pub fn new<M: map::Map>(map_buffer_size: f64, transform: &Transform2D, color_map: &ColorMap, map: &M, shader: wgpu::ShaderModuleDescriptor, render_state: &RenderState) -> Self {                
        // Create the fields
        let view = GPUMapView::new(map.get_size(), map_buffer_size, transform);
        let hex_buffers = GPUMapHexBuffers::new(render_state);
        let data_buffers = GPUMapDataBuffers::new(&view, map, render_state);
        let uniforms = GPUMapUniforms::new(color_map, render_state);
        let pipelines = GPUMapPipelines::new(shader, render_state);

        Self {
            view,
            hex_buffers,
            data_buffers,
            uniforms,
            pipelines,
        }
    }

    /// Renders the entire scene
    /// 
    /// # Parameters
    /// 
    /// render_state: The render state to use for rendering
    /// 
    /// # Errors
    /// 
    /// See RenderError for a description of the errors
    pub fn draw(&self, render_state: &RenderState) -> Result<(), RenderError> {
        // Set the transform
        self.uniforms.write_transform(&self.view.transform, render_state);

        // Get the current view
        let output_texture = render_state.get_surface().get_current_texture()?;
        let view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Setup the load ops
        let load_op_clear = wgpu::LoadOp::Clear(wgpu::Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        });
        let load_op_load = wgpu::LoadOp::Load;

        // Do the rendering
        self.draw_single(DrawMode::Fill, load_op_clear, &view, render_state);
        self.draw_single(DrawMode::Outline, load_op_load, &view, render_state);

        // Show to screen
        output_texture.present();

        Ok(())
    }

    /// Draws either the filling or the edge
    /// 
    /// # Parameters
    /// 
    /// mode: The draw mode, either filling or edge
    /// 
    /// load_op: The load operation for either resetting or added to existing drawings
    /// 
    /// render_state: The render state to use for rendering
    fn draw_single(&self, mode: DrawMode, load_op: wgpu::LoadOp<wgpu::Color>, view: &wgpu::TextureView, render_state: &RenderState) {
        // Create the encoder
        let mut encoder = render_state.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Set the draw mode to fill
        self.uniforms.write_draw_mode(mode, render_state);

        // Initialize the render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                timestamp_writes: None,
                occlusion_query_set: None,
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: wgpu::StoreOp::Store,
                    }
                })],
                depth_stencil_attachment: None,
            });

            // Set the pipeline for fill
            self.pipelines.set(mode, &mut render_pass);

            // Set the main uniforms
            self.uniforms.set(&mut render_pass);

            // Set the vertex buffer
            let vertex_count = self.hex_buffers.set(mode, &mut render_pass);

            // Draw everything
            self.data_buffers.draw(self.view.transform.get_center(), vertex_count, &mut render_pass, render_state);
        }

        // Submit
        render_state.get_queue().submit(std::iter::once(encoder.finish()));
    }
}

impl GPUMapView {
    /// Creates a new GPUMapView
    /// 
    /// # Parameters
    /// 
    /// size: The size of a chunk
    /// 
    /// map_buffer_size: The ratio of the loaded size over the rendered size, used to load more than needed to avoid constant reloading
    /// 
    /// transform: The initial transform from world coordinates to screen coordinates
    /// 
    /// # Panics
    /// 
    /// In debug mode it panics if size == 0
    fn new(size: usize, map_buffer_size: f64, transform: &Transform2D) -> Self {
        // Make sure size is large enough
        if cfg!(debug_assertions) && size < 1 {
            panic!("size must be at least 1");
        }

        // Calculate the rendered view
        let render_view = Self::transform_to_view(transform);
        let render_view = View::new(render_view.get_center(), &(render_view.get_size() * map_buffer_size));

        // Get the center index and index size
        let index_center = Self::get_center_index(size, render_view.get_center());
        let index_size = Self::get_size_index(size, render_view.get_size());

        // Get the full view
        let load_center = Self::index_to_origin(size, &index_center);
        let load_size = Self::size_index_to_size(size, &index_size);
        let view = View::new(&load_center, &load_size);

        // Create transform
        let transform = Transform2D::scale(&Point::new(0.25, 0.25)) * transform;
        
        Self {
            size,
            view,
            map_buffer_size,
            transform,
            index_center,
            index_size,
        }
    }

    /// Converts a chunk index to the origin position of that chunk
    /// 
    /// # Parameters
    /// 
    /// size: The size of a chunk
    /// 
    /// index: The index of the chunk
    fn index_to_origin(size: usize, index: &Index) -> Point {
        let x = SQRT_3 * ((index.get_x() as f64) + (index.get_y() as f64) / 2.0) * (size as f64);
        let y = (index.get_y() as f64) * (size as f64) * 1.5;
        Point::new(x, y)
    }

    /// Calculates the visible view to be rendered using the given transform
    /// 
    /// # Parameters
    /// 
    /// transform: The transform from world coordinates to screen coordinates
    fn transform_to_view(transform: &Transform2D) -> View {
        let transform_inverse = transform.inv();
        let points: (Point, Point) = [Point::new(-1.0, -1.0), Point::new(1.0, -1.0), Point::new(1.0, 1.0), Point::new(-1.0, 1.0)].iter().map(|point| {
            transform_inverse * point
        }).fold((Point::new(f64::INFINITY, f64::INFINITY), Point::new(-f64::INFINITY, -f64::INFINITY)), |prev, next| {
            (Point::new(f64::min(prev.0.get_x(), next.get_x()), f64::min(prev.0.get_y(), next.get_y())), Point::new(f64::max(prev.1.get_x(), next.get_x()), f64::max(prev.1.get_y(), next.get_y())))
        });
        View::new(&((points.0 + points.1) * 0.5), &(points.1 - points.0).to_size())
    }

    /// Calculates the index of the closest chunk to the given point
    /// 
    /// # Parameters
    /// 
    /// size: The size of a chunk
    /// 
    /// center: The point to find the closest chunk for
    fn get_center_index(size: usize, center: &Point) -> Index {
        // Calculate some base values
        let scaled_x = center.get_x() / (SQRT_3 * (size as f64));
        let scaled_y = center.get_y() / (3.0 * (size as f64));
        let remain_x = scaled_x - scaled_x.floor();
        let remain_y = scaled_y - scaled_y.floor();
        
        // Find the indices
        let cutoff = (size as f64) / 6.0 * (2.0 * (remain_x - 0.5).abs() + 1.0);
        let index_y = if remain_y < cutoff || cutoff <= remain_y { // An even row
            ((scaled_y + 0.5).floor() as i64) * 2
        } else { // an odd row
            (scaled_y.floor() as i64) * 2 + 1
        };
        let index_x = (scaled_x - 0.5 * (index_y as f64) + 0.5).floor() as i64;
        Index::new(index_x, index_y)
    }

    /// Calculates the index size for a given view size.
    /// This size is related to the size of the loaded view
    /// 
    /// # Parameters
    /// 
    /// size: The size of a chunk
    /// 
    /// view_size: The size of the view to load, the loaded size is guarenteed to be large enough such that after snapping the center, the entire original view is still loaded
    fn get_size_index(size: usize, view_size: &Size) -> Index {
        // Find the x size
        let size_x = (view_size.get_w() / (SQRT_3 * (size as f64)) + 1.0).ceil() as i64;

        // Find the y size
        let size_y = (view_size.get_h() / (3.0 * (size as f64)) + 1.0 / 3.0).ceil() as i64;

        Index::new(size_x, size_y)
    }

    /// Converts a size index to the physical size
    /// 
    /// # Parameters
    /// 
    /// size: The size of a chunk
    /// 
    /// size_index: The size index to convert
    fn size_index_to_size(size: usize, size_index: &Index) -> Size {
        let w = (size_index.get_x() as f64) * SQRT_3 * (size as f64);
        let h = ((size_index.get_y() as f64) * 3.0 + 1.0) * (size as f64);
        Size::new(w, h)
    }
}

impl GPUMapHexBuffers {
    /// Creates a new set of hexagon buffers
    /// 
    /// # Parameters
    /// 
    /// render_state: The render state to use for rendering
    fn new(render_state: &RenderState) -> Self {
        // Create the vertices
        let vertices = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Vertex Buffer"),
            contents: bytemuck::cast_slice(&Vertex::vertices_hex()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create the indices for the bulk
        let indices_bulk = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Bulk Index Buffer"),
            contents: bytemuck::cast_slice(&Vertex::indices_bulk_hex()),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create the indices for the edges
        let indices_edge = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Edge Index Buffer"),
            contents: bytemuck::cast_slice(&Vertex::indices_edge_hex()),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertices,
            indices_bulk,
            indices_edge,
        }
    }

    /// Sets the hexagon vertex information for the given render pass
    /// 
    /// # Parameters
    /// 
    /// mode: The render mode currently active
    /// 
    /// render_pass: The render pass to set the vertex info for
    fn set<'a>(&'a self, mode: DrawMode, render_pass: &mut wgpu::RenderPass<'a>) -> u32 {
        // Set the vertex buffer
        render_pass.set_vertex_buffer(0, self.vertices.slice(..));

        // Set the index buffer and return the number of indices
        match mode {
            DrawMode::Fill => {
                render_pass.set_index_buffer(self.indices_bulk.slice(..), wgpu::IndexFormat::Uint16);
                12
            }
            DrawMode::Outline => {
                render_pass.set_index_buffer(self.indices_edge.slice(..), wgpu::IndexFormat::Uint16);
                7
            }
        }
    }
}

impl GPUMapDataBuffers {
    /// Creates a new set of data buffersfor the entire loaded map
    /// 
    /// # Parameters
    /// 
    /// view_data: The view data for this GPUMap
    /// 
    /// map: The map to render
    /// 
    /// render_state: The render state to use for rendering
    fn new<M: map::Map>(view_data: &GPUMapView, map: &M, render_state: &RenderState) -> Self {
        // Get the size
        let size = map.get_size();

        // Get locations for the chunk
        let locations_chunk: Vec<Vertex> = [Vertex::from_point(&Point::new(0.0, 0.0))]
            .into_iter()
            .chain((1..size)
            .map(|layer| {
                (0..6)
                    .map(|slice| {
                        let (layer_dir, pos_dir) = match slice {
                            0 => (Point::new(0.5 * SQRT_3, 0.5), Point::new(-0.5 * SQRT_3, 0.5)),
                            1 => (Point::new(0.0, 1.0), Point::new(-0.5 * SQRT_3, -0.5)),
                            2 => (Point::new(-0.5 * SQRT_3, 0.5), Point::new(0.0, -1.0)),
                            3 => (Point::new(-0.5 * SQRT_3, -0.5), Point::new(0.5 * SQRT_3, -0.5)),
                            4 => (Point::new(0.0, -1.0), Point::new(0.5 * SQRT_3, 0.5)),
                            _ => (Point::new(0.5 * SQRT_3, -0.5), Point::new(0.0, 1.0)),
                        };
                        (0..layer)
                            .map(|pos| {
                                Vertex::from_point(&(layer_dir * (layer as f64) + pos_dir * (pos as f64)))
                            })
                            .collect::<Vec<Vertex>>()
                            .into_iter()
                    })
                    .flatten()
                    .collect::<Vec<Vertex>>()
            })
            .flatten())
            .collect();
        let locations_chunk = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer Chunk"),
            contents: bytemuck::cast_slice(&locations_chunk),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Get locations for the edges
        let locations_edge_0: Vec<Vertex> = (1..size).map(|pos| {
            let start_pos = Point::new(0.5 * SQRT_3, 0.5) * (size as f64);
            let dir = Point::new(-0.5 * SQRT_3, 0.5);
            Vertex::from_point(&(start_pos + dir * (pos as f64)))
        }).collect();
        let locations_edge_0 = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer Edge 0"),
            contents: bytemuck::cast_slice(&locations_edge_0),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let locations_edge_1: Vec<Vertex> = (1..size).map(|pos| {
            let start_pos = Point::new(0.0, 1.0) * (size as f64);
            let dir = Point::new(-0.5 * SQRT_3, -0.5);
            Vertex::from_point(&(start_pos + dir * (pos as f64)))
        }).collect();
        let locations_edge_1 = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer Edge 1"),
            contents: bytemuck::cast_slice(&locations_edge_1),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let locations_edge_2: Vec<Vertex> = (1..size).map(|pos| {
            let start_pos = Point::new(-0.5 * SQRT_3, 0.5) * (size as f64);
            let dir = Point::new(0.0, -1.0);
            Vertex::from_point(&(start_pos + dir * (pos as f64)))
        }).collect();
        let locations_edge_2 = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer Edge 2"),
            contents: bytemuck::cast_slice(&locations_edge_2),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        // Get locations for the vertices
        let locations_vertex_0 = vec![Vertex::from_point(&(Point::new(-0.5 * SQRT_3, 0.5) * (size as f64)))];
        let locations_vertex_0 = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer Vertex 0"),
            contents: bytemuck::cast_slice(&locations_vertex_0),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let locations_vertex_1 = vec![Vertex::from_point(&(Point::new(-0.5 * SQRT_3, -0.5) * (size as f64)))];
        let locations_vertex_1 = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer Vertex 1"),
            contents: bytemuck::cast_slice(&locations_vertex_1),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_center = &view_data.index_center;
        let index_size = &view_data.index_size;

        // Get the chunks to load
        let origins_chunk = (-index_size.get_y()..(index_size.get_y() + 1))
            .map(|index_y| {
                ((-(index_size.get_x() + index_y) / 2)..((index_size.get_x() - index_y) / 2 + 1))
                    .map(|index_x| {
                        Index::new(index_x + index_center.get_x(), index_y + index_center.get_y())
                    })
                    .collect::<Vec<Index>>()
                    .into_iter()
            })
            .flatten();

        // Get the edges to load
        let origins_edge_0 = (-index_size.get_y()..index_size.get_y())
            .map(|index_y| {
                ((-(index_size.get_x() + index_y) / 2)..((index_size.get_x() + 1 - index_y) / 2))
                    .map(|index_x| {
                        Index::new(index_x + index_center.get_x(), index_y + index_center.get_y())
                    })
                    .collect::<Vec<Index>>().into_iter()
            })
            .flatten();
        let origins_edge_1 = (-index_size.get_y()..index_size.get_y())
            .map(|index_y| {
                ((-(index_size.get_x() - 1 + index_y) / 2)..((index_size.get_x() - index_y) / 2 + 1))
                    .map(|index_x| {
                        Index::new(index_x + index_center.get_x(), index_y + index_center.get_y())
                    })
                    .collect::<Vec<Index>>().into_iter()
            })
            .flatten();
        let origins_edge_2 = (-index_size.get_y()..(index_size.get_y() + 1))
            .map(|index_y| {
                ((-(index_size.get_x() - 1 + index_y) / 2)..((index_size.get_x() + 1 - index_y) / 2 + 1))
                    .map(|index_x| {
                        Index::new(index_x + index_center.get_x(), index_y + index_center.get_y())
                    })
                    .collect::<Vec<Index>>().into_iter()
            })
            .flatten();

        // Get the vertices to load
        let origins_vertex_0 = origins_edge_2.clone();
        let origins_vertex_1 = origins_edge_2.clone();

        // Setup the chunk buffer
        let buffer_chunk = origins_chunk.map(|origin_index| {
            GPUMapDataBuffer::new(&map.get_chunk(&origin_index).get_id(), &GPUMapView::index_to_origin(size, &origin_index), render_state)
        }).collect();
        
        // Setup the edge buffers
        let buffer_edge_0 = origins_edge_0.map(|origin_index| {
            GPUMapDataBuffer::new(&map.get_edge_right(&origin_index).get_id(), &GPUMapView::index_to_origin(size, &origin_index), render_state)
        }).collect();
        let buffer_edge_1 = origins_edge_1.map(|origin_index| {
            GPUMapDataBuffer::new(&map.get_edge_left(&origin_index).get_id(), &GPUMapView::index_to_origin(size, &origin_index), render_state)
        }).collect();
        let buffer_edge_2 = origins_edge_2.map(|origin_index| {
            GPUMapDataBuffer::new(&map.get_edge_vertical(&origin_index).get_id(), &GPUMapView::index_to_origin(size, &origin_index), render_state)
        }).collect();

        // Setup the vertex buffers
        let buffer_vertex_0 = origins_vertex_0.map(|origin_index| {
            GPUMapDataBuffer::new(&map.get_vertex_top(&origin_index).get_id(), &GPUMapView::index_to_origin(size, &origin_index), render_state)
        }).collect();
        let buffer_vertex_1 = origins_vertex_1.map(|origin_index| {
            GPUMapDataBuffer::new(&map.get_vertex_bottom(&origin_index).get_id(), &GPUMapView::index_to_origin(size, &origin_index), render_state)
        }).collect();

        // Package all of the data
        let chunk = GPUMapDataBufferList::new(locations_chunk, buffer_chunk);
        let edges = [
            GPUMapDataBufferList::new(locations_edge_0, buffer_edge_0),
            GPUMapDataBufferList::new(locations_edge_1, buffer_edge_1),
            GPUMapDataBufferList::new(locations_edge_2, buffer_edge_2),
        ];
        let vertices = [
            GPUMapDataBufferList::new(locations_vertex_0, buffer_vertex_0),
            GPUMapDataBufferList::new(locations_vertex_1, buffer_vertex_1),
        ];

        Self {
            chunk,
            edges,
            vertices,
        }
    }

    /// Draws all of the visible tiles
    /// 
    /// # Parameters
    /// 
    /// origin: The origin of the screen in world coordinates
    /// 
    /// vertex_count: The number of vertices for the hexagon (changes depending on DrawMode)
    /// 
    /// render_pass: The render pass to draw to
    /// 
    /// render_state: The render state to use for rendering
    fn draw<'a>(&'a self, origin: &Point, vertex_count: u32, render_pass: &mut wgpu::RenderPass<'a>, render_state: &RenderState) {
        // Draw the chunks
        self.chunk.draw(origin, vertex_count, render_pass, render_state);

        // Draw the edges
        self.edges.iter().for_each(|edge| {
            edge.draw(origin, vertex_count, render_pass, render_state);
        });

        // Draw the vertices
        self.vertices.iter().for_each(|vertex| {
            vertex.draw(origin, vertex_count, render_pass, render_state);
        });
    }
}

impl GPUMapDataBufferList {
    /// Creates a new list of data buffers
    /// 
    /// # Parameters
    /// 
    /// locations: The locations of each tile relative to the chunk origin
    /// 
    /// buffers: All the instances of this group of tiles
    fn new(locations: wgpu::Buffer, buffers: Vec<GPUMapDataBuffer>) -> Self {
        Self {
            locations,
            buffers,
        }
    }

    /// Draws all of the visible instances
    /// 
    /// # Parameters
    /// 
    /// origin: The origin of the screen in world coordinates
    /// 
    /// vertex_count: The number of vertices for the hexagon (changes depending on DrawMode)
    /// 
    /// render_pass: The render pass to draw to
    /// 
    /// render_state: The render state to use for rendering
    fn draw<'a>(&'a self, origin: &Point, vertex_count: u32, render_pass: &mut wgpu::RenderPass<'a>, render_state: &RenderState) {
        // Set the locations
        render_pass.set_vertex_buffer(1, self.locations.slice(..));

        // Render the buffers            
        for buffer in &self.buffers {                
            // Set the specific data
            buffer.set(origin, render_pass, render_state);
            
            // Draw the hexagons
            render_pass.draw_indexed(0..vertex_count, 0, 0..buffer.size as u32);            
        }
    }
}

impl GPUMapDataBuffer {
    /// Creates a new data buffer for the map data for one region of the map
    /// 
    /// # Parameters
    /// 
    /// ids: A slice of all the ids for this buffer, these can be overwritten later with set_ids
    /// 
    /// render_state: The render state to use for rendering
    fn new(ids: &[u32], origin: &Point, render_state: &RenderState) -> Self {
        // Set the size and origin
        let size = ids.len();
        let origin_data = *origin;
        
        let ids = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer"),
            contents: bytemuck::cast_slice(ids),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Get a buffer for the origin
        let origin = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Origin Buffer"),
            size: mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group for origin
        let bind_group_origin = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Origin"),
            layout: &Self::bind_group_layout(render_state),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: origin.as_entire_binding(),
                },
            ]
        });

        Self {
            size,
            origin_data,
            ids,
            origin,
            bind_group_origin,
        }
    }

    /// Sets the id buffer
    /// 
    /// # Parameters
    /// 
    /// ids: The ids of each tile
    /// 
    /// render_state: The render state to use for rendering
    /// 
    /// # Panics
    /// 
    /// In debug mode this panics if the length of ids is not equal to the original length of ids given when initialized
    fn set_ids(&self, ids: &[u32], render_state: &RenderState) {
        // Make sure the ids slice is the correct length
        if cfg!(debug_assertions) && ids.len() != self.size {
            panic!("The length of id must be {:?} but received {:?}", self.size, ids.len());
        }

        render_state.get_queue().write_buffer(&self.ids, 0, bytemuck::cast_slice(&ids));
    }

    /// Sets all of the gpu data for this data buffer
    /// 
    /// Warning: Must not be called multiple times per render pass with different origins as only the last origin will be used for all renderings
    /// 
    /// # Parameters
    /// 
    /// origin: The location of the center of the screen in world coordinates
    /// 
    /// render_pass: The render pass to draw to
    /// 
    /// render_state: The render state to use for rendering
    fn set<'a>(&'a self, origin: &Point, render_pass: &mut wgpu::RenderPass<'a>, render_state: &RenderState) {
        // Set the origin buffer
        render_state.get_queue().write_buffer(&self.origin, 0, bytemuck::cast_slice(&Vertex::from_points(&[self.origin_data - *origin])));
        
        // Set instance data
        render_pass.set_vertex_buffer(2, self.ids.slice(..));

        // Set the origin
        render_pass.set_bind_group(1, &self.bind_group_origin, &[]);
    }

    /// Creates the bind group layout for a data buffer
    /// 
    /// # Parameters
    /// 
    /// render_state: The render state to use for rendering
    fn bind_group_layout(render_state: &RenderState) -> wgpu::BindGroupLayout {
        render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Origin Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}

impl GPUMapUniforms {
    /// Creates a new set of uniforms for the gpu
    /// 
    /// # Parameters
    /// 
    /// color_map: The color map to use when rendering the tiles
    /// 
    /// render_state: The render state to use for rendering
    fn new(color_map: &ColorMap, render_state: &RenderState) -> Self {
        // Create transform buffer
        let center_transform = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Buffer"),
            size: (mem::size_of::<f32>() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create draw mode buffer
        let draw_mode = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Mode Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create the color map
        let color_map = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Map Buffer"),
            contents: bytemuck::cast_slice(color_map.get_data()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create bind group for the uniforms
        let bind_group = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Main"),
            layout: &Self::bind_group_layout(render_state),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: center_transform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: draw_mode.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: color_map.as_entire_binding(),
                },
            ]
        });

        Self {
            center_transform,
            draw_mode,
            _color_map: color_map,
            bind_group,
        }
    }

    /// Update the transform, this must be run once before the first rendering as the transform is not initialized
    /// 
    /// # Parameters
    /// 
    /// transform: The transform from world coordinates to screen coordinates
    /// 
    /// render_state: The render state to use for rendering
    fn write_transform(&self, transform: &Transform2D, render_state: &RenderState) {
        render_state.get_queue().write_buffer(&self.center_transform, 0, bytemuck::cast_slice(&transform.get_data_center_transform()));
    }

    /// Update the draw mode, this must be run once before the first rendering as the transform is not initialized
    /// 
    /// # Parameters
    /// 
    /// mode: The draw mode to tell the shader if it is drawing edges or filling
    /// 
    /// render_state: The render state to use for rendering
    fn write_draw_mode(&self, mode: DrawMode, render_state: &RenderState) {
        render_state.get_queue().write_buffer(&self.draw_mode, 0, bytemuck::cast_slice(&[mode.get_data()]));
    }

    /// Binds the uniforms to the given render pass
    /// 
    /// # Parameters
    /// 
    /// render_pass: The render pass to draw to
    fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_bind_group(0, &self.bind_group, &[]);
    }

    /// Creates the bind group layout for a set of uniforms
    /// 
    /// # Parameters
    /// 
    /// render_state: The render state to use for rendering
    fn bind_group_layout(render_state: &RenderState) -> wgpu::BindGroupLayout {
        render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Main Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}

impl GPUMapPipelines {
    /// Creates a new set of render pipelines
    /// 
    /// # Parameters
    /// 
    /// shader: The shader program to use
    /// 
    /// render_state: The render state to use for rendering
    fn new(shader: wgpu::ShaderModuleDescriptor, render_state: &RenderState) -> Self {
        // Create shader
        let shader = render_state.get_device().create_shader_module(shader);

        // Create render pipeline
        let layout = render_state.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout Descriptor"),
            bind_group_layouts: &[&GPUMapUniforms::bind_group_layout(render_state), &GPUMapDataBuffer::bind_group_layout(render_state)],
            push_constant_ranges: &[],
        });

        // Create the fill pipeline
        let fill = render_state.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline Fill"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc_hex(),
                    Vertex::desc_location(),
                    Vertex::desc_id(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_state.get_config().format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create the outline pipeline
        let outline = render_state.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline Fill"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc_hex(),
                    Vertex::desc_location(),
                    Vertex::desc_id(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_state.get_config().format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            fill,
            outline,
        }
    }

    /// Sets the correct pipeline for the render pass
    /// 
    /// # Parameters
    /// 
    /// mode: The draw mode, this determines what pipeline to use
    /// 
    /// render_pass: The render pass to draw to
    fn set<'a>(&'a self, mode: DrawMode, render_pass: &mut wgpu::RenderPass<'a>) {
        match mode {
            DrawMode::Fill => render_pass.set_pipeline(&self.fill),
            DrawMode::Outline => render_pass.set_pipeline(&self.outline),
        }
    }
}

/// Describes a single vertex in the gpu
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// The position in the plane
    position: [f32; 2],
}

impl Vertex {
    /// Gets the memory description of a hex vertex
    fn desc_hex() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ]
        }
    }

    /// Gets the memory description of a location
    fn desc_location() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ]
        }
    }

    /// Gets the memory description of a id
    fn desc_id() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
            ]
        }
    }
    
    /// Generates vertices for one hexagon
    fn vertices_hex() -> [Self; 6] {
        [
            Self { position: [       INV_SQRT_3 as f32,  0.0] },
            Self { position: [ 0.5 * INV_SQRT_3 as f32,  0.5] },
            Self { position: [-0.5 * INV_SQRT_3 as f32,  0.5] },
            Self { position: [-      INV_SQRT_3 as f32,  0.0] },
            Self { position: [-0.5 * INV_SQRT_3 as f32, -0.5] },
            Self { position: [ 0.5 * INV_SQRT_3 as f32, -0.5] },
        ]
    }

    /// Generates indices for the vertices for the bulk of a hexagon
    const fn indices_bulk_hex() -> [u16; 12] {
        [
            2, 3, 4,
            2, 4, 5,
            1, 2, 5,
            0, 1, 5,
        ]
    }

    /// Generates indices for the vertices for the edge of a hexagon
    const fn indices_edge_hex() -> [u16; 7] {
        [0, 1, 2, 3, 4, 5, 0]
    }

    /// Converts a slice of points to a vector of Vertex objects
    /// 
    /// # Parameters
    /// 
    /// points: The points to convert
    fn from_points(points: &[Point]) -> Vec<Self> {
        points.iter().map(|point| Self { position: [point.get_x() as f32, point.get_y() as f32] }).collect()
    }

    /// Converts a single point to a vertex
    fn from_point(point: &Point) -> Self {
        Self { position: [point.get_x() as f32, point.get_y() as f32] }
    }
}

/// Describes if rendering should be done on the filling or outline of hexagons
#[derive(Copy, Clone, Debug)]
enum DrawMode {
    Fill,
    Outline,
}

impl DrawMode {
    /// Retrieves the code for the gpu for this mode
    fn get_data(&self) -> u32 {
        match *self {
            Self::Fill => 0,
            Self::Outline => 1,
        }
    }
}

/// Used to describe errors when rendering
#[derive(Error, Debug, Clone)]
pub enum RenderError {
    /// The surface texture could not be retrieved
    #[error("Unable to get surface texture: {:?}", .0)]
    SurfaceTexture(wgpu::SurfaceError),
}

impl From<wgpu::SurfaceError> for RenderError {
    fn from(err: wgpu::SurfaceError) -> RenderError {
        RenderError::SurfaceTexture(err)
    }
}
