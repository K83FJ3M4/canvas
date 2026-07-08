@group(0) @binding(0)
var<storage, read_write> keysA: array<u32>;

@group(0) @binding(1)
var<storage, read_write> keysB: array<u32>;

@group(0) @binding(2)
var<storage, read_write> histograms: array<array<u32, 16>>;

@group(0) @binding(3)
var<uniform> shift: u32;

var<workgroup> localKeys: array<u32, 256>;
var<workgroup> localHistograms: array<array<u32, 16>, 16>;

struct Builtin {
    @builtin(global_invocation_id) globalIndex: vec3u,
    @builtin(local_invocation_id) localIndex: vec3u,
    @builtin(workgroup_id) workgroupIndex: vec3u,
    @builtin(num_workgroups) numWorkgroups: vec3u,
}

@compute @workgroup_size(256)
fn countKeys(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.x;
    let workgroupIndex = builtin.workgroupIndex.x;
    let globalOffset = globalIndex - localIndex;
    let keyCount = keyArrayLength() - globalOffset;

    localKeys[localIndex] = readKey(globalIndex);
    workgroupBarrier();

    var count = 0u;
    let lane = localIndex & 15u;
    let tile = localIndex >> 4;
    let tileStart = tile * 16u;
    let tileEnd = min(tileStart + 16u, keyCount);

    for (var i = tileStart; i < tileEnd; i++) {
        let key = (localKeys[i] >> shift) & 15u;
        count += u32(key == lane);
    }

    localHistograms[tile][lane] = count;
    workgroupBarrier();

    if (localIndex >= 16u) { return; }

    count = 0u;
    for (var i = 0u; i < 16u; i++) {
        count += localHistograms[i][localIndex];
    }
    histograms[workgroupIndex][localIndex] = count;
}

@compute @workgroup_size(256)
fn countHistograms(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let numWorkgroups = builtin.numWorkgroups.x;
    let workgroupIndex = builtin.workgroupIndex.x;
    let histogramRange = getHistogramRange(numWorkgroups);

    let lane = localIndex & 15u;
    let tile = localIndex >> 4;

    var count = 0u;
    let globalOffset = histogramRange.x;
    let workgroupOffset = globalOffset + workgroupIndex * 256u;
    let tileStart = workgroupOffset + tile * 16u;
    let tileEnd = min(tileStart + 16u, histogramRange.y);

    for (var i = tileStart; i < tileEnd; i++) {
        count += histograms[i][lane];
    }

    localHistograms[tile][lane] = count;
    workgroupBarrier();

    if (localIndex >= 16u) { return; }

    count = 0u;
    for (var i = 0u; i < 16u; i++) {
        count += localHistograms[i][localIndex];
    }
    let targetIndex = histogramRange.y + workgroupIndex;
    histograms[targetIndex][localIndex] = count;
}

fn getHistogramRange(numWorkgroups: u32) -> vec2u {
    var offset = 0u;
    var sourceCount = divCeil256(keyArrayLength());
    var outputCount = divCeil256(sourceCount);
    while (numWorkgroups < outputCount) {
        offset += sourceCount;
        sourceCount = outputCount;
        outputCount = divCeil256(sourceCount);
    }

    return vec2(offset, offset + sourceCount);
}

fn readKey(index: u32) -> u32 {
    if (bool(shift & 4u)) {
        let last = arrayLength(&keysA) - 1u;
        return keysA[min(last, index)];
    } else {
        let last = arrayLength(&keysB) - 1u;
        return keysB[min(last, index)];
    }
}

fn keyArrayLength() -> u32 {
    if (bool(shift & 4u)) {
        return arrayLength(&keysA);
    } else {
        return arrayLength(&keysB);
    }
}

fn divCeil256(num: u32) -> u32 {
    return ((num - 1u) >> 8u) + 1u;
}