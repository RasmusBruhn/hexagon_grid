// Vertex shader
@group(0) @binding(0)
var<uniform> origin: vec2<f32>;

@group(0) @binding(1)
var<uniform> transform: mat2x2<f32>;

@group(1) @binding(0)
var<uniform> chunk_offset: vec2<f32>;

@group(2) @binding(0)
var<uniform> draw_mode: u32;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) id: u32,
};

@vertex
fn vs_main(
    @location(0) hex_offset: vec2<f32>,
    @location(1) location: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.id = 0u;
    var pos = transform * (location + hex_offset + chunk_offset - origin);
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    if draw_mode == 0u {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
}