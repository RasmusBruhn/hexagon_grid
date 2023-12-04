use super::{
    INV_SQRT_3, SQRT_3, START_COLOR, END_COLOR,
    types::{Point, Size, Index, View, Transform2D},
    render::RenderState,
    map,
};
use std::{mem, f64};
use wgpu::util::DeviceExt;
use thiserror::Error;

/// The representation of a map in the gpu allowing for rendering
pub struct GPUMap {
    /// The currently loaded view, when the visible region goes outside of this then a reload is needed
    view: View,
    /// The amount of the map to load relative to the shown area, must be at least 1
    map_buffer_size: f64,
    /// The transform of the screen
    transform: Transform2D,
    /// The buffer for all the loaded chunks
    buffer_chunk: Vec<Buffer>,
    /// The buffer for all the loaded edges
    buffer_edge: [Vec<Buffer>; 3],
    /// The buffer for all the loaded vetices
    buffer_vertex: [Vec<Buffer>; 2],
    /// The transformation matrix
    buffer_center_transform: wgpu::Buffer,
    /// The draw mode buffer
    buffer_draw_mode: wgpu::Buffer,
    /// The color map buffer
    _buffer_color_map: wgpu::Buffer,
    /// The bind group for the center transform and the draw mode
    bind_group: wgpu::BindGroup,
    /// The render pipeline for filling
    pipeline_fill: wgpu::RenderPipeline,
    /// The render pipeline for the outline
    pipeline_outline: wgpu::RenderPipeline,
}


impl GPUMap {
    pub fn new<M: map::Map>(size: usize, map_buffer_size: f64, transform: &Transform2D, map: &M, render_state: &RenderState) -> Self {
        if cfg!(debug_assertions) && size < 1 {
            panic!("size must be at least 1");
        }

        // Calculate the rendered view
        let render_view = Self::transform_to_view(transform);
        let render_view = View::new(*render_view.get_center(), *render_view.get_size() * map_buffer_size);

        // Get the center index and index size
        let index_center = Self::get_center_index(size, render_view.get_center());
        let index_size = Self::get_size_index(size, render_view.get_size());

        // Get the full view
        let load_center = Self::chunk_index_to_pos(size, &index_center);
        let load_size = Self::size_index_to_size(size, &index_size);
        let view = View::new(load_center, load_size);

        // Get the chunks to load
        let origins_chunk = (-index_size.get_y()..(index_size.get_y() + 1)).map(|index_y| {
            ((-(index_size.get_x() + index_y) / 2)..((index_size.get_x() - index_y) / 2 + 1)).map(|index_x| {
                Index::new(index_x, index_y)
            }).collect::<Vec<Index>>().into_iter()
        }).flatten();

        // Get the edges to load
        let origins_edge_0 = (-index_size.get_y()..(index_size.get_y() + 1)).map(|index_y| {
            ((-(index_size.get_x() - 1 + index_y) / 2)..((index_size.get_x() + 1 - index_y) / 2 + 1)).map(|index_x| {
                Index::new(index_x, index_y)
            }).collect::<Vec<Index>>().into_iter()
        }).flatten();
        let origins_edge_1 = (-index_size.get_y()..index_size.get_y()).map(|index_y| {
            ((-(index_size.get_x() - 1 + index_y) / 2)..((index_size.get_x() - index_y) / 2 + 1)).map(|index_x| {
                Index::new(index_x, index_y)
            }).collect::<Vec<Index>>().into_iter()
        }).flatten();
        let origins_edge_2 = (-index_size.get_y()..index_size.get_y()).map(|index_y| {
            ((-(index_size.get_x() + index_y) / 2)..((index_size.get_x() + 1 - index_y) / 2)).map(|index_x| {
                Index::new(index_x, index_y)
            }).collect::<Vec<Index>>().into_iter()
        }).flatten();

        // Get the vertices to load
        let origins_vertex_0 = origins_edge_0.clone();
        let origins_vertex_1 = origins_edge_0.clone();

        // Get locations for the chunk
        let locations_chunk: Vec<Point> = [Point::new(0.0, 0.0)].into_iter().chain((1..size).map(|layer| {
            (0..6).map(|slice| {
                let (layer_dir, pos_dir) = match slice {
                    0 => (Point::new(0.5 * SQRT_3, 0.5), Point::new(-0.5 * SQRT_3, 0.5)),
                    1 => (Point::new(0.0, 1.0), Point::new(-0.5 * SQRT_3, -0.5)),
                    2 => (Point::new(-0.5 * SQRT_3, 0.5), Point::new(0.0, -1.0)),
                    3 => (Point::new(-0.5 * SQRT_3, -0.5), Point::new(0.5 * SQRT_3, -0.5)),
                    4 => (Point::new(0.0, -1.0), Point::new(0.5 * SQRT_3, 0.5)),
                    _ => (Point::new(0.5 * SQRT_3, -0.5), Point::new(0.0, 1.0)),
                };
                (0..layer).map(|pos| {
                    layer_dir * (layer as f64) + pos_dir * (pos as f64)
                }).collect::<Vec<Point>>().into_iter()
            }).flatten().collect::<Vec<Point>>()
        }).flatten()).collect();

        // Get locations for the edges
        let locations_edge_0: Vec<Point> = (1..size).map(|pos| {
            let start_pos = Point::new(-0.5 * SQRT_3 * (size as f64), -0.5 * (size as f64));
            let dir = Point::new(0.0, 1.0);
            start_pos + dir * (pos as f64)
        }).collect();
        let locations_edge_1: Vec<Point> = (1..size).map(|pos| {
            let start_pos = Point::new(-0.5 * SQRT_3 * (size as f64), 0.5 * (size as f64));
            let dir = Point::new(0.5 * SQRT_3, 0.5);
            start_pos + dir * (pos as f64)
        }).collect();
        let locations_edge_2: Vec<Point> = (1..size).map(|pos| {
            let start_pos = Point::new(0.0, size as f64);
            let dir = Point::new(0.5 * SQRT_3, -0.5);
            start_pos + dir * (pos as f64)
        }).collect();
        
        // Get locations for the vertices
        let locations_vertex_0 = vec![Point::new(-0.5 * SQRT_3 * (size as f64), -0.5 * (size as f64))];
        let locations_vertex_1 = vec![Point::new(-0.5 * SQRT_3 * (size as f64), 0.5 * (size as f64))];

        // Setup the chunk buffer
        let buffer_chunk = origins_chunk.map(|origin_index| {
            Buffer::new(&locations_chunk, &map.get_chunk(origin_index).get_id(), &Self::chunk_index_to_pos(size, &origin_index), render_state)
        }).collect();
        
        // Setup the edge buffers
        let buffer_edge_0 = origins_edge_0.map(|origin_index| {
            Buffer::new(&locations_edge_0, &map.get_edge_vertical(origin_index).get_id(), &Self::chunk_index_to_pos(size, &origin_index), render_state)
        }).collect();
        let buffer_edge_1 = origins_edge_1.map(|origin_index| {
            Buffer::new(&locations_edge_1, &map.get_edge_left(origin_index).get_id(), &Self::chunk_index_to_pos(size, &origin_index), render_state)
        }).collect();
        let buffer_edge_2 = origins_edge_2.map(|origin_index| {
            Buffer::new(&locations_edge_2, &map.get_edge_right(origin_index).get_id(), &Self::chunk_index_to_pos(size, &origin_index), render_state)
        }).collect();
        let buffer_edge = [buffer_edge_0, buffer_edge_1, buffer_edge_2];

        // Setup the vertex buffers
        let buffer_vertex_0 = origins_vertex_0.map(|origin_index| {
            Buffer::new(&locations_vertex_0, &map.get_vertex_bottom(origin_index).get_id(), &Self::chunk_index_to_pos(size, &origin_index), render_state)
        }).collect();
        let buffer_vertex_1 = origins_vertex_1.map(|origin_index| {
            Buffer::new(&locations_vertex_1, &map.get_vertex_top(origin_index).get_id(), &Self::chunk_index_to_pos(size, &origin_index), render_state)
        }).collect();
        let buffer_vertex = [buffer_vertex_0, buffer_vertex_1];

        // Create transform
        let transform = Transform2D::scale(&Point::new(0.25, 0.25)) * transform;
        
        // Create transform buffer
        let buffer_center_transform = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Buffer"),
            size: (mem::size_of::<f32>() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create draw mode buffer
        let buffer_draw_mode = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Mode Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create the color map
        let color_map: Vec<[f32; 4]> = (0..size + 1).map(|id| {
            let fraction = (id as f32) / (size as f32);
            [
                fraction * (END_COLOR[0] - START_COLOR[0]) + START_COLOR[0],
                fraction * (END_COLOR[1] - START_COLOR[1]) + START_COLOR[1],
                fraction * (END_COLOR[2] - START_COLOR[2]) + START_COLOR[2],
                fraction * (END_COLOR[3] - START_COLOR[3]) + START_COLOR[3],
            ]
        }).collect();
        let buffer_color_map = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Map Buffer"),
            contents: bytemuck::cast_slice(&color_map),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create bind group for transform and mode
        let bind_group_layout = render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });
        let bind_group = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Main"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_center_transform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_draw_mode.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_color_map.as_entire_binding(),
                },
            ]
        });

        // The layout for the Buffer bind groups
        let bind_group_origin_layout = render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

        // Create shader
        let shader = render_state.get_device().create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // Create render pipeline
        let pipeline_layout = render_state.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout Descriptor"),
            bind_group_layouts: &[&bind_group_layout, &bind_group_origin_layout],
            push_constant_ranges: &[],
        });
        let pipeline_fill = render_state.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline Fill"),
            layout: Some(&pipeline_layout),
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
        let pipeline_outline = render_state.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline Fill"),
            layout: Some(&pipeline_layout),
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
            view,
            map_buffer_size,
            transform,
            buffer_chunk,
            buffer_edge,
            buffer_vertex,
            buffer_center_transform,
            buffer_draw_mode,
            _buffer_color_map: buffer_color_map,
            bind_group,
            pipeline_fill,
            pipeline_outline,
        }
    }

    pub fn render(&self, render_state: &RenderState) -> Result<(), RenderError> {
        // Set the transform
        render_state.get_queue().write_buffer(&self.buffer_center_transform, 0, bytemuck::cast_slice(&self.transform.get_data_center_transform()));

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
        self.render_single(DrawMode::Fill, load_op_clear, &view, render_state);
        self.render_single(DrawMode::Outline, load_op_load, &view, render_state);

        // Show to screen
        output_texture.present();

        Ok(())
    }

    fn render_single(&self, mode: DrawMode, load_op: wgpu::LoadOp<wgpu::Color>, view: &wgpu::TextureView, render_state: &RenderState) {
        // Create the encoder
        let mut encoder = render_state.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Set the draw mode to fill
        render_state.get_queue().write_buffer(&self.buffer_draw_mode, 0, bytemuck::cast_slice(&[mode.get_data()]));

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
            match mode {
                DrawMode::Fill => render_pass.set_pipeline(&self.pipeline_fill),
                DrawMode::Outline => render_pass.set_pipeline(&self.pipeline_outline)
            };
            
            // Set the main uniforms
            render_pass.set_bind_group(0, &self.bind_group, &[]);

            // Render the chunks
            for chunk in &self.buffer_chunk {
                chunk.draw(&self.transform.get_center(), mode, &mut render_pass, render_state)
            }

            // Render the edges
            for edge_list in &self.buffer_edge {
                for edge in edge_list {
                    edge.draw(&self.transform.get_center(), mode, &mut render_pass, render_state)
                }
            }

            // Render the vertices
            for vertex_list in &self.buffer_vertex {
                for vertex in vertex_list {
                    vertex.draw(&self.transform.get_center(), mode, &mut render_pass, render_state)
                }    
            }
        }

        // Submit
        render_state.get_queue().submit(std::iter::once(encoder.finish()));
    }

    fn chunk_index_to_pos(size: usize, coord: &Index) -> Point {
        let x = SQRT_3 * ((coord.get_x() as f64) + (coord.get_y() as f64) / 2.0) * (size as f64);
        let y = (coord.get_y() as f64) * (size as f64) * 1.5;
        Point::new(x, y)
    }

    fn transform_to_view(transform: &Transform2D) -> View {
        let transform_inverse = transform.inv();
        let points: (Point, Point) = [Point::new(-1.0, -1.0), Point::new(1.0, -1.0), Point::new(1.0, 1.0), Point::new(-1.0, 1.0)].iter().map(|point| {
            transform_inverse * point
        }).fold((Point::new(f64::INFINITY, f64::INFINITY), Point::new(-f64::INFINITY, -f64::INFINITY)), |prev, next| {
            (Point::new(f64::min(prev.0.get_x(), next.get_x()), f64::min(prev.0.get_y(), next.get_y())), Point::new(f64::max(prev.1.get_x(), next.get_x()), f64::max(prev.1.get_y(), next.get_y())))
        });
        View::new((points.0 + points.1) * 0.5, (points.1 - points.0).to_size())
    }

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

    fn get_size_index(size: usize, view_size: &Size) -> Index {
        // Find the x size
        let size_x = (view_size.get_w() / (SQRT_3 * (size as f64)) + 1.0).ceil() as i64;

        // Find the y size
        let size_y = (view_size.get_h() / (3.0 * (size as f64)) + 1.0 / 3.0).ceil() as i64;

        Index::new(size_x, size_y)
    }

    fn size_index_to_size(size: usize, size_index: &Index) -> Size {
        let w = (size_index.get_x() as f64) * SQRT_3 * (size as f64);
        let h = ((size_index.get_y() as f64) * 3.0 + 1.0) * (size as f64);
        Size::new(w, h)
    }
}

/// A buffer which can draw a group of hexagons like a chunk or edge
/// 
/// S: The number of hexagons in one element
struct Buffer {
    /// The number of hexagons
    size: usize,
    /// The origin of the hexagons (center coordinate is: location + origin)
    origin_data: Point,
    /// The locations of each of the hexagons
    locations: wgpu::Buffer,
    /// The id of each of the hexagons
    ids: wgpu::Buffer,
    /// The buffer to write origin to the gpu
    origin: wgpu::Buffer,
    /// The bind group for the origin
    bind_group_origin: wgpu::BindGroup,
    /// The buffer holding all of the vertex data for a hexagon
    hex_vertices: wgpu::Buffer,
    /// The buffer holding all of the bulk index data for a hexagon
    hex_indices_bulk: wgpu::Buffer,
    /// The buffer holding all of the edge index data for a hexagon
    hex_indices_edge: wgpu::Buffer,
}

impl Buffer {
    /// Create a new buffer
    /// 
    /// # Parameters
    /// 
    /// locations: The locations of each hexagon
    /// 
    /// ids: The ids of each hexagon, must have same length as locations
    /// 
    /// origin: The origin
    /// 
    /// render_state: The render state to use for rendering
    fn new(locations: &[Point], ids: &[u32], origin: &Point, render_state: &RenderState) -> Self {
        if cfg!(debug_assertions) && locations.len() != ids.len() {
            panic!("The length of ids ({:?}) must be equal to that of locations ({:?})", ids.len(), locations.len());
        }
        
        // Set the size and origin
        let size = locations.len();
        let origin_data = *origin;
        
        // Create the hex buffers
        let hex_vertices = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Vertex Buffer"),
            contents: bytemuck::cast_slice(&Vertex::vertices_hex()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let hex_indices_bulk = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Bulk Index Buffer"),
            contents: bytemuck::cast_slice(&Vertex::indices_bulk_hex()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let hex_indices_edge = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Edge Index Buffer"),
            contents: bytemuck::cast_slice(&Vertex::indices_edge_hex()),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create the buffers for locations and ids
        let locations = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer"),
            contents: bytemuck::cast_slice(&Vertex::from_points(locations)),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let ids = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer"),
            contents: bytemuck::cast_slice(ids),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Get a buffer for the origin
        let origin = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Origin Buffer"),
            size: mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group for transform
        let bind_group_origin_layout = render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });
        let bind_group_origin = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Origin"),
            layout: &bind_group_origin_layout,
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
            locations,
            ids,
            origin,
            bind_group_origin,
            hex_vertices,
            hex_indices_bulk,
            hex_indices_edge,
        }
    }

    /// Sets the id buffer
    /// 
    /// # Parameters
    /// 
    /// ids: The ids of each hexagon
    /// 
    /// render_state: The render state to use for rendering
    fn set_ids(&self, ids: &[u32], render_state: &RenderState) {
        // Make sure the ids slice is the correct length
        if cfg!(debug_assertions) && ids.len() != self.size {
            panic!("The length of id must be {:?} but received {:?}", self.size, ids.len());
        }

        render_state.get_queue().write_buffer(&self.ids, 0, bytemuck::cast_slice(&ids));
    }

    /// Draws all elements
    /// 
    /// Warning: Must not be called multiple times per render pass with different origins as only the last origin will be used for all renderings
    /// 
    /// # Parameters
    /// 
    /// origin: The location of the center of the screen in world coordinates
    /// 
    /// origin_buffer: The buffer to write the origin to
    /// 
    /// render_pass: The render pass to render with
    /// 
    /// render_state: The render state to use for rendering
    fn draw<'a>(&'a self, origin: &Point, mode: DrawMode, render_pass: &mut wgpu::RenderPass<'a>, render_state: &RenderState) {
        // Set the origin buffer
        render_state.get_queue().write_buffer(&self.origin, 0, bytemuck::cast_slice(&Vertex::from_points(&[self.origin_data - *origin])));
        
        // Set the vertex buffer
        render_pass.set_vertex_buffer(0, self.hex_vertices.slice(..));

        // Set the index buffer
        let vertex_count = match mode {
            DrawMode::Fill => {
                render_pass.set_index_buffer(self.hex_indices_bulk.slice(..), wgpu::IndexFormat::Uint16);
                12
            }
            DrawMode::Outline => {
                render_pass.set_index_buffer(self.hex_indices_edge.slice(..), wgpu::IndexFormat::Uint16);
                7
            }
        };

        // Set instance data
        render_pass.set_vertex_buffer(1, self.locations.slice(..));
        render_pass.set_vertex_buffer(2, self.ids.slice(..));

        // Set the origin
        render_pass.set_bind_group(1, &self.bind_group_origin, &[]);

        // Draw the hexagons
        render_pass.draw_indexed(0..vertex_count, 0, 0..self.size as u32);            
    }
}

/// Describes a vertex within a hexagon
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
}

/// Describes if rendering should be done on the filling or outline of hexagons
#[derive(Copy, Clone, Debug)]
enum DrawMode {
    Fill,
    Outline,
}

impl DrawMode {
    fn get_data(&self) -> u32 {
        match *self {
            Self::Fill => 0,
            Self::Outline => 1,
        }
    }
}

#[derive(Error, Debug, Clone)]
pub enum RenderError {
    #[error("Unable to get surface texture: {:?}", .0)]
    SurfaceTexture(wgpu::SurfaceError),
}

impl From<wgpu::SurfaceError> for RenderError {
    fn from(err: wgpu::SurfaceError) -> RenderError {
        RenderError::SurfaceTexture(err)
    }
}
