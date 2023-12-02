use super::{
    N, SQRT_3, INV_SQRT_3,
    map,
    types::{Point, Size, Index},
    render::RenderState,
};
use wgpu::util::DeviceExt;
use thiserror::Error;
use std::rc::Rc;

/// A wrapper around a map which can draw to the screen
pub struct GPUMap {
    /// The currently loaded view, reload some chunks when the screen view goes outside of this
    //view: View,
    /// The size of the buffer of the view
    //view_buffer_size: f64,
    /// The buffer for all the loaded chunks
    buffer_chunk: Buffer<{3 * N * (N - 1) + 1}>,
    /// The buffer for all the loaded edges
    _buffer_edge: [Buffer<{N - 1}>; 3],
    /// The buffer for all the loaded vetices
    _buffer_vertex: [Buffer<1>; 2],
    /// The layout of the chunk origin bind group
    chunk_bind_group_origin_layout: Rc<wgpu::BindGroupLayout>,
    /// The transformation matrix
    _center_transform: wgpu::Buffer,
    /// The offset vector
    _origin: wgpu::Buffer,
    /// The bind group for the center transform and origin uniforms
    bind_group_transform: wgpu::BindGroup,
    /// The draw mode buffer
    draw_mode: wgpu::Buffer,
    /// The bind group for the constants
    bind_group_const: wgpu::BindGroup,
    /// The render pipeline for filling
    pipeline_fill: wgpu::RenderPipeline,
    /// The render pipeline for the outline
    pipeline_outline: wgpu::RenderPipeline,
}

impl GPUMap {
    /// Create a new gpu map
    /// 
    /// # Parameters
    /// 
    /// map: The logical map
    /// 
    /// view: The first view of the map to initially load, also defines the size
    /// 
    /// view_buffer_size: How large the buffer zone around the view should be, should be larger or equal to 1
    pub fn new<M: map::Map>(map: &M, render_state: &RenderState) -> Self {
        //if view_buffer_size < 1.0 {
        //    return Err(NewMapError::InvalidBufferSize(view_buffer_size));
        //}

        // Create the buffers
        //let min_size = *view.get_size() * view_buffer_size + Size::new(SQRT_3 * (N as f64), 2.0 * (N as f64));
        //let (size, size_buffer_chunk, size_buffer_edge, size_buffer_vertex) = Self::buffer_size(&min_size);

        // Create the bind group layout for the chunk origins
        let chunk_bind_group_origin_layout = Rc::new(render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Origin Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        }));

        // Get locations for the chunk
        let locations_chunk = {
            let mut slice: usize = 5;
            let mut pos: usize = 0;
            let mut layer: usize = 0;
            std::array::from_fn(|_| {
                let (layer_dir, pos_dir) = match slice {
                    0 => (Point::new(0.5 * SQRT_3, 0.5), Point::new(-0.5 * SQRT_3, 0.5)),
                    1 => (Point::new(0.0, 1.0), Point::new(-0.5 * SQRT_3, -0.5)),
                    2 => (Point::new(-0.5 * SQRT_3, 0.5), Point::new(0.0, -1.0)),
                    3 => (Point::new(-0.5 * SQRT_3, -0.5), Point::new(0.5 * SQRT_3, -0.5)),
                    4 => (Point::new(0.0, -1.0), Point::new(0.5 * SQRT_3, 0.5)),
                    _ => (Point::new(0.5 * SQRT_3, -0.5), Point::new(0.0, 1.0)),
                };
                let location = layer_dir * (layer as f64) + pos_dir * (pos as f64);
                pos += 1;
                if pos >= layer {
                    pos = 0;
                    slice += 1;
                    if slice >= 6 {
                        slice = 0;
                        layer += 1;
                    }
                }
                location
            })
        };

        // Setup the chunk buffer
        let buffer_chunk = Buffer::new(&locations_chunk, &[Point::new(0.0, 0.0)], chunk_bind_group_origin_layout.clone(), render_state);

        // Get locations for the edges
        let locations_edge_0 = {
            std::array::from_fn(|pos| {
                let start_pos = Point::new(-0.5 * SQRT_3 * (N as f64), -0.5 * (N as f64));
                let dir = Point::new(0.0, 1.0);
                start_pos + dir * ((pos + 1) as f64)
            })
        };
        let locations_edge_1 = {
            std::array::from_fn(|pos| {
                let start_pos = Point::new(-0.5 * SQRT_3 * (N as f64), 0.5 * (N as f64));
                let dir = Point::new(0.5 * SQRT_3, 0.5);
                start_pos + dir * ((pos + 1) as f64)
            })
        };
        let locations_edge_2 = {
            std::array::from_fn(|pos| {
                let start_pos = Point::new(0.0, N as f64);
                let dir = Point::new(0.5 * SQRT_3, -0.5);
                start_pos + dir * ((pos + 1) as f64)
            })
        };

        // Setup the edge buffers
        let buffer_edge_0 = Buffer::new(&locations_edge_0, &[Point::new(0.0, 0.0), Point::new(SQRT_3, 0.0) * (N as f64)], chunk_bind_group_origin_layout.clone(), render_state);
        let buffer_edge_1 = Buffer::new(&locations_edge_1, &[Point::new(0.0, 0.0), Point::new(0.5 * SQRT_3, -1.5) * (N as f64)], chunk_bind_group_origin_layout.clone(), render_state);
        let buffer_edge_2 = Buffer::new(&locations_edge_2, &[Point::new(0.0, 0.0), Point::new(-0.5 * SQRT_3, -1.5) * (N as f64)], chunk_bind_group_origin_layout.clone(), render_state);
        let buffer_edge = [buffer_edge_0, buffer_edge_1, buffer_edge_2];

        // Get locations for the vertices
        let locations_vertex_0 = [Point::new(-0.5 * SQRT_3, -0.5) * (N as f64)];
        let locations_vertex_1 = [Point::new(-0.5 * SQRT_3, 0.5) * (N as f64)];

        // Setup the vertex buffers
        let buffer_vertex_0 = Buffer::new(&locations_vertex_0, &[Point::new(0.0, 0.0), Point::new(SQRT_3, 0.0) * (N as f64), Point::new(0.5 * SQRT_3, 1.5) * (N as f64)], chunk_bind_group_origin_layout.clone(), render_state);
        let buffer_vertex_1 = Buffer::new(&locations_vertex_1, &[Point::new(0.0, 0.0), Point::new(0.5 * SQRT_3, -1.5) * (N as f64), Point::new(SQRT_3, 0.0) * (N as f64)], chunk_bind_group_origin_layout.clone(), render_state);
        let buffer_vertex = [buffer_vertex_0, buffer_vertex_1];

        // Fill in ids
        let data_chunk = map.get_chunk(Index::new(0, 0)).get_id();
        let data_edge_0 = [
            map.get_edge_vertical(Index::new(0, 0)).get_id(),
            map.get_edge_vertical(Index::new(1, 0)).get_id(),
        ].concat();
        let data_edge_1 = [
            map.get_edge_left(Index::new(0, 0)).get_id(),
            map.get_edge_left(Index::new(1, -1)).get_id(),
        ].concat();
        let data_edge_2 = [
            map.get_edge_right(Index::new(0, 0)).get_id(),
            map.get_edge_right(Index::new(0, -1)).get_id(),
        ].concat();
        let data_vertex_0 = [
            map.get_vertex_bottom(Index::new(0, 0)).get_id(),
            map.get_vertex_bottom(Index::new(1, 0)).get_id(),
            map.get_vertex_bottom(Index::new(0, 1)).get_id(),
        ].concat();
        let data_vertex_1 = [
            map.get_vertex_top(Index::new(0, 0)).get_id(),
            map.get_vertex_top(Index::new(1, -1)).get_id(),
            map.get_vertex_top(Index::new(1, 0)).get_id(),
        ].concat();

        buffer_chunk.set_id(data_chunk, render_state);
        buffer_edge[0].set_id(data_edge_0, render_state);
        buffer_edge[1].set_id(data_edge_1, render_state);
        buffer_edge[2].set_id(data_edge_2, render_state);
        buffer_vertex[0].set_id(data_vertex_0, render_state);
        buffer_vertex[1].set_id(data_vertex_1, render_state);

        // Create transform
        let center_transform = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::cast_slice(&[0.2 as f32, 0.0 as f32, 0.0 as f32, 0.2 as f32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create offset
        let origin = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Buffer"),
            size: (std::mem::size_of::<f32>() * 2) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group for transform
        let bind_group_transform_layout = render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Transform Layout"),
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
        let bind_group_transform = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Transform"),
            layout: &bind_group_transform_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: origin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: center_transform.as_entire_binding(),
                },
            ]
        });

        // Create offset
        let draw_mode = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Mode Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Setup bind group for the constants
        let bind_group_const_layout = render_state.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Const Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group_const = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Const"),
            layout: &bind_group_const_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: draw_mode.as_entire_binding(),
                },
            ]
        });

        // Create shader
        let shader = render_state.get_device().create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // Create render pipeline
        let pipeline_layout = render_state.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout Descriptor"),
            bind_group_layouts: &[&bind_group_transform_layout, &chunk_bind_group_origin_layout, &bind_group_const_layout],
            push_constant_ranges: &[],
        });
        let pipeline_fill = render_state.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline Fill"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc_vertex(),
                    Vertex::desc_instance(),
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
                    Vertex::desc_vertex(),
                    Vertex::desc_instance(),
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
            buffer_chunk,
            _buffer_edge: buffer_edge,
            _buffer_vertex: buffer_vertex,
            chunk_bind_group_origin_layout,
            _center_transform: center_transform,
            _origin: origin,
            bind_group_transform,
            draw_mode,
            bind_group_const,
            pipeline_fill,
            pipeline_outline,
        }
    }

    pub fn render(&self, render_state: &RenderState) -> Result<(), RenderError> {
        // Get the current view
        let output_texture = render_state.get_surface().get_current_texture()?;
        let view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create the encoder
        let mut encoder = render_state.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Set the constants
        render_state.get_queue().write_buffer(&self.draw_mode, 0, bytemuck::cast_slice(&[0 as u32]));

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
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    }
                })],
                depth_stencil_attachment: None,
            });

            // Set the pipeline for fill
            render_pass.set_pipeline(&self.pipeline_fill);
            
            // Set the constants
            render_pass.set_bind_group(2, &self.bind_group_const, &[]);

            // Set the transform
            render_pass.set_bind_group(0, &self.bind_group_transform, &[]);

            // Render the chunks
            self.buffer_chunk.draw(DrawMode::Fill, &mut render_pass);

            // Render the edges
            self._buffer_edge[0].draw(DrawMode::Fill, &mut render_pass);
            //self._buffer_edge[1].draw(DrawMode::Fill, &mut render_pass);
            //self._buffer_edge[2].draw(DrawMode::Fill, &mut render_pass);
        }

        // Submit
        render_state.get_queue().submit(std::iter::once(encoder.finish()));

        // Create the encoder
        let mut encoder = render_state.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Set the constants for outline
        render_state.get_queue().write_buffer(&self.draw_mode, 0, bytemuck::cast_slice(&[1 as u32]));

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
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }
                })],
                depth_stencil_attachment: None,
            });

            // Set the pipeline for outline
            render_pass.set_pipeline(&self.pipeline_outline);

            // Set the constants
            render_pass.set_bind_group(2, &self.bind_group_const, &[]);

            // Set the transform
            render_pass.set_bind_group(0, &self.bind_group_transform, &[]);

            // Render the chunks
            self.buffer_chunk.draw(DrawMode::Outline, &mut render_pass);
        }
        
        // Submit
        render_state.get_queue().submit(std::iter::once(encoder.finish()));

        // Show to screen
        output_texture.present();

        Ok(())
    }

    /// Calculates the number of chunks, edges, and vertices to load.
    /// Returns 0: The size it actually loaded, larger than view_size, 1: Number of chunks,
    /// 2: Number of each type of edge, 3: Number of each type of vertex
    ///
    /// # Parameters
    /// 
    /// view_size: The size of the view to load
    fn _buffer_size(view_size: &Size) -> (Size, usize, [usize; 3], [usize; 2]) {
        // Calculate the number of number of chunk positions to load in x and y
        let x = (view_size.get_w() / (SQRT_3 * (N as f64))).ceil() as usize;
        let y = ((view_size.get_h() - (N as f64)) / (3.0 * N as f64)).ceil() as usize;
        
        // Calculate height and width for the vens and odds
        let w_even = 2 * (x / 2) + 1;
        let w_odd = 2 * ((x + 1) / 2);
        let h_even = 2 * (y / 2) + 1;
        let h_odd = 2 * ((y + 1) / 2);

        // Calculate the number of chunks
        let size_chunk = h_even * w_even + h_odd * w_odd;

        // Calculate the number of each type of edge
        let size_edge_0 = h_even * w_odd + h_odd * w_even;
        let size_edge_1 = ((h_even - 1) + h_odd) * x;
        let size_edge_2 = size_edge_1;
        let size_edge = [size_edge_0, size_edge_1, size_edge_2];

        // Calculate the number of vertices
        let size_vertex_0 =  h_even * w_odd + h_odd * w_even;
        let size_vertex_1 = size_vertex_0;
        let size_vertex = [size_vertex_0, size_vertex_1];

        // Calculate the size of the loaded view
        let size_w = (x as f64) * SQRT_3 * (N as f64);
        let size_h = (y as f64) * 3.0 * (N as f64) + (N as f64);
        let size = Size::new(size_w, size_h);

        (
            size,
            size_chunk,
            size_edge,
            size_vertex,
        )
    }

    fn _index_to_pos_chunk(coord: &Index) -> Point {
        let y = (coord.get_x() as f64) * ((3 * N - 1) as f64) / 2.0;
        let x = ((coord.get_x() + 2 * coord.get_y()) as f64) * (N as f64) * 0.5 * SQRT_3;
        Point::new(x, y)
    }    
}

/// A buffer to hold several chunks of data
/// 
/// S: The number of hexagons in one element
struct Buffer<const S: usize> {
    /// The buffer holding all of the vertex data for a hexagon
    hex_vertices: wgpu::Buffer,
    /// The buffer holding all of the bulk index data for a hexagon
    hex_indices_bulk: wgpu::Buffer,
    /// The buffer holding all of the edge index data for a hexagon
    hex_indices_edge: wgpu::Buffer,
    /// The locations of each of the hexagons in one element relative the origin
    locations: wgpu::Buffer,
    /// The origin of each of the elements
    _origin: wgpu::Buffer,
    /// The layout of the origin bind group
    _bind_group_origin_layout: Rc<wgpu::BindGroupLayout>,
    /// The bind group for the origin
    bind_group_origin: wgpu::BindGroup,
    /// The id buffers for each element
    id: wgpu::Buffer,
    /// The currently saved number of elements
    size: usize,
}

impl<'a, const S: usize> Buffer<S> {
    /// Create a new buffer
    /// 
    /// # Parameters
    /// 
    /// locations: The locations of each hexagon within one element
    /// 
    /// origins: The origin of each element
    /// 
    /// device: The device for the gpu to create buffers
    fn new(locations: &[Point; S], origins: &[Point], bind_group_origin_layout: Rc<wgpu::BindGroupLayout>, render_state: &RenderState) -> Self {
        let hex_vertices = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Vertex Buffer"),
            contents: bytemuck::cast_slice(&Vertex::hex_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let hex_indices_bulk = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Bulk Index Buffer"),
            contents: bytemuck::cast_slice(&Vertex::hex_indices_bulk()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let hex_indices_edge = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hex Edge Index Buffer"),
            contents: bytemuck::cast_slice(&Vertex::hex_indices_edge()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let locations = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Location Buffer"),
            contents: bytemuck::cast_slice(&Vertex::from_points(locations)),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let size = origins.len();

        let origin = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Origin Buffer"),
            contents: bytemuck::cast_slice(&VertexUniform::from_points(origins)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group for transform
        let bind_group_origin = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Origin"),
            layout: bind_group_origin_layout.as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: origin.as_entire_binding(),
                },
            ]
        });

        let id = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Id Buffer {:?}", size)),
            size: (std::mem::size_of::<u32>() * S * size) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            hex_vertices,
            hex_indices_bulk,
            hex_indices_edge,
            locations,
            _origin: origin,
            _bind_group_origin_layout: bind_group_origin_layout,
            bind_group_origin,
            id,
            size,
        }
    }

    /// Changes the number of elements
    /// 
    /// # Parameters
    /// 
    /// origins: The origin of each element
    /// 
    /// device: The device for the gpu to create buffers
    pub fn _resize(&mut self, origins: &[Point], render_state: &RenderState) {
        self.size = origins.len();

        self.id = render_state.get_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Id Buffer {:?}", self.size)),
            size: (std::mem::size_of::<u32>() * S * self.size) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self._origin = render_state.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Origin Buffer"),
            contents: bytemuck::cast_slice(&Vertex::from_points(origins)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group for origin
        self.bind_group_origin = render_state.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Origin"),
            layout: self._bind_group_origin_layout.as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self._origin.as_entire_binding(),
                },
            ]
        });
    }

    /// Set the id buffer for each of the elements
    /// 
    /// # Parameters
    /// 
    /// id: The ids of each hex in each element, vector should be as long as the number of entities
    /// 
    /// queue: The command queue to write to the buffer
    fn set_id(&self, id: Vec<u32>, render_state: &RenderState) {
        if cfg!(debug_assertions) && id.len() != S * self.size {
            panic!("The length of id must be {:?} but received {:?}", self.size, S * id.len());
        }

        render_state.get_queue().write_buffer(&self.id, 0, bytemuck::cast_slice(&id));
    }

    /// Sets the origin and ids for a single element
    /// 
    /// # Parameters
    /// 
    /// index: The index of the element to set, this must be smaller than the size
    /// 
    /// origin: The new origin of this element
    /// 
    /// id: The id buffer for this element
    /// 
    /// queue: The command queue to write to the buffer
    fn _set_element(&mut self, index: usize, origin: &Point, id: &[u32; S], render_state: &RenderState) {
        if cfg!(debug_assertions) && index >= self.size {
            panic!("The index must be smaller than {:?} but received {:?}", self.size, index);
        }
        
        render_state.get_queue().write_buffer(
            &self.id, 
            (std::mem::size_of::<u32>() * S * index) as u64, 
            bytemuck::cast_slice(id)
        );

        render_state.get_queue().write_buffer(
            &self._origin,
            (std::mem::size_of::<Vertex>() * index) as u64,
            bytemuck::cast_slice(&Vertex::from_points(std::slice::from_ref(origin))),
        );
    }

    pub fn _get_size(&self) -> usize {
        self.size
    }

    /// Draws all elements
    /// 
    /// # Parameters
    /// 
    /// origin: The location of the center of the screen in world coordinates
    /// 
    /// origin_buffer: The buffer to write the origin to
    /// 
    /// render_pass: The render pass to render with
    /// 
    /// queue: The queue to usue to write to buffers
    fn draw(&'a self, mode: DrawMode, render_pass: &mut wgpu::RenderPass<'a>) {
        // Set vertex buffer
        render_pass.set_vertex_buffer(0, self.hex_vertices.slice(..));

        // Set the index
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
        //render_pass.set_vertex_buffer(2, self.id.slice(..));

        for index in 0..self.size {
            // Set the local offset
            render_pass.set_bind_group(1, &self.bind_group_origin, &[(index * std::mem::size_of::<VertexUniform>()) as wgpu::DynamicOffset]);

            // Draw the element
            render_pass.draw_indexed(0..vertex_count, 0, (S * index) as u32..(S * (index + 1)) as u32);            
        }
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
    /// Gets the memory description of a vertex
    fn desc_vertex() -> wgpu::VertexBufferLayout<'static> {
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

    /// Gets the memory description of a vertex
    fn desc_instance() -> wgpu::VertexBufferLayout<'static> {
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
    
    /// Generates vertices for one hexagon
    /// 
    /// # Parameters
    /// 
    /// center: The center of the hexagon
    /// 
    /// id: The colour id of the hexagon
    pub fn hex_vertices() -> [Self; 6] {
        [
            Self { position: [INV_SQRT_3 as f32, 0.0] },
            Self { position: [0.5 * INV_SQRT_3 as f32, 0.5] },
            Self { position: [-0.5 * INV_SQRT_3 as f32, 0.5] },
            Self { position: [-INV_SQRT_3 as f32, 0.0] },
            Self { position: [-0.5 * INV_SQRT_3 as f32, -0.5] },
            Self { position: [0.5 * INV_SQRT_3 as f32, -0.5] },
        ]
    }

    /// Generates indices for the vertices for the bulk of a hexagon
    pub const fn hex_indices_bulk() -> [u16; 12] {
        [
            2, 3, 4,
            2, 4, 5,
            1, 2, 5,
            0, 1, 5,
        ]
    }

    /// Generates indices for the vertices for the edge of a hexagon
    pub const fn hex_indices_edge() -> [u16; 7] {
        [0, 1, 2, 3, 4, 5, 0]
    }

    pub fn from_points(points: &[Point]) -> Vec<Self> {
        points.iter().map(|point| Self { position: [point.get_x() as f32, point.get_y() as f32] }).collect()
    }
}

/// Describes a vertex which can be used in a uniform array
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexUniform {
    /// The position in the plane
    position: [f32; 2],
    /// Just for alignment
    _fill: [u64; 31],
}

impl VertexUniform {
    pub fn from_points(points: &[Point]) -> Vec<Self> {
        points.iter().map(|point| Self { position: [point.get_x() as f32, point.get_y() as f32], _fill: [0; 31] }).collect()
    }
}

enum DrawMode {
    Fill,
    Outline,
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