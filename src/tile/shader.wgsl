@group(0) @binding(1)
var<storage, read_write> lists: array<u32>;

@group(0) @binding(2)
var<storage, read_write> listRanges: array<array<u32, 2>>;

@group(0) @binding(3)
var<storage, read_write> triangles: array<Triangle>;

@group(0) @binding(4)
var<storage, read_write> params: Params;

var<workgroup> listHeads: array<ListHead, 64>;

struct Params {
    sampleFractionBits: u32
}

struct Triangle {
    clockwise: u32,
    material: u32,
    a: vec2u,
    b: vec2u,
    c: vec2u
}

struct ListHead {
    offset: u32,
    length: u32,
    item: u32
}

struct Builtin {
    @builtin(workgroup_id) workgroupIndex: vec3u,
    @builtin(global_invocation_id) globalIndex: vec3<u32>,
    @builtin(local_invocation_id) localIndex: vec3<u32>
}

@compute @workgroup_size(16, 16)
fn main(builtin: Builtin) {
    let workgroupIndex = builtin.workgroupIndex.xy;
    let localIndex = builtin.localIndex.y * 16u + builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.xy;
    let size = textureDimensions(output);

    let bits = params.sampleFractionBits;
    let sample = (globalIndex << vec2(bits)) + vec2(1u << (bits - 1u));

    loadListHeads(localIndex, workgroupIndex);
    sortListHeads(localIndex);

    let uv = vec2<f32>(builtin.localIndex.xy) / 15.0;
    var color = vec4<f32>(uv.x, uv.y, 0.5, 1.0);

    var stencil = 0u;
    for (var i = 0u; i < arrayLength(&triangles); i++) {
        let triangle = triangles[i];
        let insideA = insideEdge(triangle.a, sample);
        let insideB = insideEdge(triangle.b, sample);
        let insideC = insideEdge(triangle.c, sample);

        if (insideA && insideB && insideC) {
            if (bool(triangle.clockwise)) {
                stencil -= 1u;
            } else {
                stencil += 1u;
            }
        }
    }

    if (bool(stencil & 1u)) {
        color =  vec4(0.0, 0.0, 0.0, 1.0);
    }

    while (listHeads[0].length > 0u) {
        let item = nextItem(localIndex);
        //color = vec4(0.0, 0.0, 0.0, 1.0);
    }


    if (all(globalIndex < size)) {
        textureStore(output, globalIndex, color); 
    }
}

fn insideEdge(edge: vec2u, sample: vec2u) -> bool {
    let a = bitcast<i32>(edge.x << 16u) >> 16u;
    let b = bitcast<i32>(edge.x) >> 16u;
    let c = bitcast<i32>(edge.y);
    return (a * i32(sample.x) + b * i32(sample.y) + c) >= 0i;
}

fn nextItem(localIndex: u32) -> u32 {
    var base = listHeads[0];
    let item = base.item;
    base.length -= 1u;
    base.offset += 1u;
    if (base.length > 0u) {
        base.item = lists[base.offset];
    }

    var nextHead: ListHead;
    var replaced = false;
    var replaces = false;

    if (localIndex + 1u < 64u) {
        nextHead = listHeads[localIndex + 1u];
        replaced = listHeadLessThan(nextHead, base);
    } 

    if (localIndex < 64u) {
        let current = listHeads[localIndex];
        replaces = listHeadLessThan(current, base);
    }

    workgroupBarrier();

    let leadingList = localIndex == 0u;
    if (!replaced && (replaces || leadingList)) {
        listHeads[localIndex] = base;
    } else if (replaced) {
        listHeads[localIndex] = nextHead;
    }

    workgroupBarrier();
    return item;
}

fn sortListHeads(localIndex: u32) {
    for (var size = 2u; size <= 64u; size = size << 1u) {
        for (var stride = size >> 1u; stride != 0u; stride = stride >> 1u) {
            let other = localIndex ^ stride;
            if (other > localIndex && localIndex < 64u) {
                let ascending = (localIndex & size) == 0u;
                let indexA = select(localIndex, other, ascending);
                let indexB = select(other, localIndex, ascending);
                let a = listHeads[indexA];
                let b = listHeads[indexB];

                if (listHeadLessThan(a, b)) {
                    listHeads[indexA] = b;
                    listHeads[indexB] = a;
                }
            }

            workgroupBarrier();
        }
    }
}

fn loadListHeads(localIndex: u32, tile: vec2u) {
    
    let level = localIndex >> 2u;
    let phase = localIndex & 3u;

    let fractionBits = min(params.sampleFractionBits, 11u);
    let levelCount = 12u - fractionBits;

    let firstLevel = level == 0u;
    let lastLevel = level == levelCount - 1u;
    var head = emptyListHead();

    if (firstLevel && phase == 0u) {
        head = newListHead(0u);
    } else if (lastLevel && phase == 0u) {
        let levelOffset = getLevelOffset(level);
        let tileOffset = tile.y * (1u << level) + tile.x;
        head = newListHead(levelOffset + tileOffset);
    } else if (level < levelCount && !firstLevel && !lastLevel) {
        let levelOffset = getLevelOffset(level);
        let phaseSize = getPhaseSize(level);
        let phaseOffset = levelOffset + phaseSize * phase;

        let shiftAmount = levelCount - 1u - level;
        let halfShift = 1u << (shiftAmount - 1u);

        let shiftX = phase & 1u;
        let shiftY = phase >> 1u;

        let x = (tile.x + shiftX * halfShift) >> shiftAmount;
        let y = (tile.y + shiftY * halfShift) >> shiftAmount;

        let width = getPhaseWidth(level);
        let tileOffset = y * width + x;
        head = newListHead(phaseOffset + tileOffset);
    }

    if (localIndex < 64u) {
        listHeads[localIndex] = head;
    }
    workgroupBarrier();
}

fn getLevelOffset(level: u32) -> u32 {
    let alternating = 0x55555555u;
    let lowBits = (1u << (2u * level)) - 1u;
    let geometric = (alternating & lowBits) - 1u;

    let squareSum = geometric << 2u;
    let linearSum = (1u << (level + 4u)) - 32u;
    let constantSum = 16u * (level - 1u);
    let offset = squareSum + linearSum + constantSum;
    return offset + 1u;
}

fn newListHead(rangeIndex: u32) -> ListHead {
    var item = 0u;
    let range = listRanges[rangeIndex];
    let offset = range[0];
    let length = range[1] - range[0];
    if (length != 0) { item = lists[offset]; }
    return ListHead(offset, length, item);
}

fn listHeadLessThan(a: ListHead, b: ListHead) -> bool {
    let valid = vec2(a.length, b.length) > vec2(0u);
    return (a.item < b.item && all(valid))
        || (valid.x && !valid.y);    
}

fn getPhaseSize(level: u32) -> u32 {
    return (1u << (2u * level))
        + (1u << (level + 2u))
        + 4u;
}

fn emptyListHead() -> ListHead {
    return ListHead(0u, 0u, 0u);
}

fn getPhaseWidth(level: u32) -> u32 {
    return (1u << level) + 2u;
}