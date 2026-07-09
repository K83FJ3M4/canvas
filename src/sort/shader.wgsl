@group(0) @binding(0)
var<storage, read_write> keysA: array<u32>;

@group(0) @binding(1)
var<storage, read_write> keysB: array<u32>;

@group(0) @binding(2)
var<storage, read_write> histograms: array<array<u32, 16>>;

@group(0) @binding(3)
var<uniform> shift: u32;

var<workgroup> localKeys: array<u32, 256>;
var<workgroup> offsetHistogram: array<u32, 16>;
var<workgroup> countHistograms: array<array<u32, 16>, 16>;

struct Builtin {
    @builtin(global_invocation_id) globalIndex: vec3u,
    @builtin(local_invocation_id) localIndex: vec3u,
    @builtin(workgroup_id) workgroupIndex: vec3u,
    @builtin(num_workgroups) numWorkgroups: vec3u,
}

@compute @workgroup_size(256)
fn scanKeys(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.x;
    let workgroupIndex = builtin.workgroupIndex.x;
    let blockStart = globalIndex - localIndex;
    let keyCount = min(256u, keyArrayLength() - blockStart);
    let valid = localIndex < keyCount; 

    let key = loadLocalKeys(globalIndex);
    workgroupBarrier();

    let bucket = localIndex & 15u;
    let tile = localIndex >> 4u;
    let tileStart = tile * 16u;
    let tileEnd = min(tileStart + 16u, keyCount);

    var count = 0u;
    for (var i = tileStart; i < tileEnd; i++) {
        let digit = (localKeys[i] >> shift) & 15u;
        count += u32(digit == bucket);
    }

    countHistograms[tile][bucket] = count;
    workgroupBarrier();

    if (localIndex < 16u) {
        var sum = 0u;
        for (var i = 0u; i < 16u; i++) {
            let value = countHistograms[i][localIndex];
            countHistograms[i][localIndex] = sum;
            sum += value;
        }
    }

    workgroupBarrier();

    if (valid) {
        let digit = (key >> shift) & 15u;

        var localRank = countHistograms[tile][digit];
        for (var i = tileStart; i < localIndex; i++) {
            let otherDigit = (localKeys[i] >> shift) & 15u;
            localRank += u32(otherDigit == digit);
        }

        let dst = histograms[workgroupIndex][digit] + localRank;
        writeKey(dst, key);
    }
}

@compute @workgroup_size(256)
fn scanHistograms(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let numWorkgroups = builtin.numWorkgroups.x;
    let workgroupIndex = builtin.workgroupIndex.x;
    let histogramRange = getHistogramRange(numWorkgroups);

    if (localIndex < 16u) {
        let index = histogramRange.y + workgroupIndex;
        offsetHistogram[localIndex] = histograms[index][localIndex];
    }

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

    countHistograms[tile][lane] = count;
    workgroupBarrier();

    if (localIndex < 16u) {
        count = offsetHistogram[localIndex];
        for (var i = 0u; i < 16u; i++) {
            let value = countHistograms[i][localIndex];
            countHistograms[i][localIndex] = count;
            count += value;
        }
    }

    workgroupBarrier();
    count = countHistograms[tile][lane];

    for (var i = tileStart; i < tileEnd; i++) {
        let value = histograms[i][lane];
        histograms[i][lane] = count;
        count += value;
    }
}

@compute @workgroup_size(16)
fn initOffsetHistogram(builtin: Builtin) {
    let index = getHistogramRange(1).y;
    let localIndex = builtin.localIndex.x;
    offsetHistogram[localIndex] = histograms[index][localIndex];
    workgroupBarrier();

    var sum = 0u;
    if (localIndex == 0u) {
        for (var i = 0u; i < 16u; i++) {
            let count = offsetHistogram[i];
            offsetHistogram[i] = sum;
            sum += count;
        }
    }

    workgroupBarrier();
    histograms[index][localIndex] = offsetHistogram[localIndex];
}

@compute @workgroup_size(256)
fn countKeys(builtin: Builtin) {
    let localIndex = builtin.localIndex.x;
    let globalIndex = builtin.globalIndex.x;
    let workgroupIndex = builtin.workgroupIndex.x;
    let globalOffset = globalIndex - localIndex;
    let keyCount = keyArrayLength() - globalOffset;

    loadLocalKeys(globalIndex);
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

    countHistograms[tile][lane] = count;
    workgroupBarrier();

    if (localIndex >= 16u) { return; }

    count = 0u;
    for (var i = 0u; i < 16u; i++) {
        count += countHistograms[i][localIndex];
    }
    histograms[workgroupIndex][localIndex] = count;
}

@compute @workgroup_size(256)
fn mergeHistograms(builtin: Builtin) {
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

    countHistograms[tile][lane] = count;
    workgroupBarrier();

    if (localIndex >= 16u) { return; }

    count = 0u;
    for (var i = 0u; i < 16u; i++) {
        count += countHistograms[i][localIndex];
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

fn loadLocalKeys(globalIndex: u32) -> u32 {
    let validA = globalIndex < arrayLength(&keysA);
    let validB = globalIndex < arrayLength(&keysB);
    let localIndex = globalIndex & 255u;
    let sourceB = bool(shift & 4u);
    var key = 0u;

    if (validA && !sourceB) {
        key = keysA[globalIndex];
    } else if (validB && sourceB) {
        key = keysB[globalIndex];
    }

    localKeys[localIndex] = key;
    return key;
}

fn writeKey(index: u32, key: u32) {
    if (!bool(shift & 4u)) {
        keysB[index] = key;
    } else {
        keysA[index] = key;
    }
}

fn keyArrayLength() -> u32 {
    if (!bool(shift & 4u)) {
        return arrayLength(&keysA);
    } else {
        return arrayLength(&keysB);
    }
}

fn divCeil256(num: u32) -> u32 {
    return ((num - 1u) >> 8u) + 1u;
}