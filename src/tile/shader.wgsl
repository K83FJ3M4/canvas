@group(0) @binding(1)
var<storage, read_write> lists: array<u32>;

@group(0) @binding(2)
var<storage, read_write> listRanges: array<array<u32, 2>>;

struct Builtin {
    @builtin(global_invocation_id) globalIndex : vec3<u32>,
    @builtin(local_invocation_id) localIndex: vec3<u32>
}

@compute @workgroup_size(16, 16)
fn main(builtin: Builtin) {
    let localIndex = builtin.localIndex.xy;
    let globalIndex = builtin.globalIndex.xy;
    let size = textureDimensions(output);

    let uv = vec2<f32>(localIndex) / 15.0;
    var color = vec4<f32>(uv.x, uv.y, 0.5, 1.0); 

    if (all(globalIndex < size)) {
        textureStore(output, globalIndex, color); 
    }
}