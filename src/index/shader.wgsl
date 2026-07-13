@group(0) @binding(0)
var<storage, read_write> sortedListKeys: array<u32>;

@group(0) @binding(1)
var<storage, read_write> listRanges: array<array<u32, 2>>;

struct Builtin {
    @builtin(global_invocation_id) globalIndex: vec3u
}

@compute @workgroup_size(256)
fn main(builtin: Builtin) {
    let globalIndex = builtin.globalIndex.x;
    let length = arrayLength(&sortedListKeys);
    if (globalIndex >= length) { return; }
    let currentList = sortedListKeys[globalIndex];
    let valid = currentList != ~0u;

    if (globalIndex == 0u) {
        if (valid) {
            listRanges[currentList][0] = globalIndex;
        }
    } else {
        let previousList = sortedListKeys[globalIndex - 1u]; 
        if (currentList != previousList) {
            listRanges[previousList][1] = globalIndex;
            if (valid) {
                listRanges[currentList][0] = globalIndex;
            }
        }
    }

    if (globalIndex == length - 1u && valid) {
        listRanges[currentList][1] = length;
    }
}