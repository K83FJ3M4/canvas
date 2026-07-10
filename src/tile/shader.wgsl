@group(0) @binding(1)
var<storage, read_write> lists: array<u32>;

@group(0) @binding(2)
var<storage, read_write> listRanges: array<array<u32, 2>>;

var<workgroup> listHeads: array<ListHead, 64>;
var<workgroup> listCount: u32;

struct ListHead {
    offset: u32,
    length: u32,
    item: u32
}

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

fn mergeLists(localIndex: u32) {
    var base = listHeads[0];
    var baseEmpty = base.length == 0u;
    let hasNext = localIndex + 1u < listCount;

    if (!baseEmpty) {
        base.item = lists[base.offset];
    }

    base.length -= u32(!baseEmpty);
    base.offset += u32(!baseEmpty);

    var nextHead: ListHead;
    var replaced = false;
    var replaces = false;

    if (hasNext) {
        nextHead = listHeads[localIndex + 1u];
    }

    if (hasNext && !baseEmpty) {
        replaced = nextHead.item < base.item; 
    }

    if (localIndex < listCount && !baseEmpty) {
        let current = listHeads[localIndex];
        replaces = current.item < base.item;
    }

    workgroupBarrier();

    let leadingList = localIndex == 0u && !baseEmpty;
    if (!replaced && (replaces || leadingList)) {
        listHeads[localIndex] = base;
    } else if (replaced || (baseEmpty && hasNext)) {
        listHeads[localIndex] = nextHead;
    }

    if (baseEmpty && localIndex == 0u) {
        listCount -= 1u;
    }

    workgroupBarrier();
}