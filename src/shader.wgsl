// Vertex shader
@group(0) @binding(0)
var<uniform> transform: mat2x2<f32>;

@group(0) @binding(1)
var<uniform> draw_mode: u32;

@group(0) @binding(2)
var<storage, read> color_map: array<vec4<f32>>;

@group(1) @binding(0)
var<uniform> origin: vec2<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) id: u32,
};

@vertex
fn vs_main(
    @location(0) hex_offset: vec2<f32>,
    @location(1) location: vec2<f32>,
    @location(2) id: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.id = id;
    var pos = transform * (origin + location + hex_offset);
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    if (draw_mode == 0u) {
        return color_map[in.id];
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
}