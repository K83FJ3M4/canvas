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
var<workgroup> activeHistogramRange: vec2u;

struct Builtin {
    @builtin(global_invocation_id) globalIndex: vec3u,
    @builtin(num_workgroups) numWorkgroups: vec3u,
}

@compute @workgroup_size(256)
fn scanKeys(builtin: Builtin) {
    let globalIndex = builtin.globalIndex.x;

    clearOffsetHistogram(globalIndex);
    loadLocalKeys(globalIndex);
    workgroupBarrier();

    countTileKeys(globalIndex);
    workgroupBarrier(); 

    scanLocalHistograms(globalIndex);
    workgroupBarrier();

    scatterKey(globalIndex);
}

@compute @workgroup_size(256)
fn scanHistograms(builtin: Builtin) {
    let globalIndex = builtin.globalIndex.x;
    let numWorkgroups = builtin.numWorkgroups.x;

    getHistogramRange(globalIndex, numWorkgroups);
    workgroupBarrier();

    loadOffsetHistogram(globalIndex);
    countHistogramTile(globalIndex);
    workgroupBarrier();

    scanLocalHistograms(globalIndex);
    workgroupBarrier();

    scanHistogramTile(globalIndex);
}

@compute @workgroup_size(16)
fn initOffsetHistogram(builtin: Builtin) {
    let globalIndex = builtin.globalIndex.x;

    getHistogramRange(globalIndex, 1u);
    workgroupBarrier();

    loadOffsetHistogram(globalIndex);
    workgroupBarrier();

    scanOffsetHistogram(globalIndex);

    workgroupBarrier();
    storeOffsetHistogram(globalIndex);
}

@compute @workgroup_size(256)
fn countKeys(builtin: Builtin) {
    let globalIndex = builtin.globalIndex.x;

    loadLocalKeys(globalIndex);
    workgroupBarrier();

    countTileKeys(globalIndex);
    workgroupBarrier();

    storeKeyHistogram(globalIndex);
}

@compute @workgroup_size(256)
fn mergeHistograms(builtin: Builtin) {
    let globalIndex = builtin.globalIndex.x;
    let numWorkgroups = builtin.numWorkgroups.x;

    getHistogramRange(globalIndex, numWorkgroups);
    workgroupBarrier();

    countHistogramTile(globalIndex);
    workgroupBarrier();

    storeMergedHistogram(globalIndex);
}

fn getHistogramRange(globalIndex: u32, numWorkgroups: u32) {
    if (getLocalIndex(globalIndex) != 0u) { return; }

    var offset = 0u;
    var sourceCount = divCeil256(keyArrayLength());
    var outputCount = divCeil256(sourceCount);
    while (numWorkgroups < outputCount) {
        offset += sourceCount;
        sourceCount = outputCount;
        outputCount = divCeil256(sourceCount);
    }

    activeHistogramRange = vec2u(offset, offset + sourceCount);
}

fn clearOffsetHistogram(globalIndex: u32) {
    let localIndex = getLocalIndex(globalIndex);
    if (localIndex < 16u) {
        offsetHistogram[localIndex] = 0u;
    }
}

fn scanLocalHistograms(globalIndex: u32) -> u32 {
    let localIndex = getLocalIndex(globalIndex);
    if (localIndex >= 16u) { return 0u; }

    var sum = offsetHistogram[localIndex];
    for (var i = 0u; i < 16u; i++) {
        let value = countHistograms[i][localIndex];
        countHistograms[i][localIndex] = sum;
        sum += value;
    }

    return sum;
}

fn scanOffsetHistogram(globalIndex: u32) {
    if (globalIndex != 0u) { return; }

    var sum = 0u;
    for (var i = 0u; i < 16u; i++) {
        let count = offsetHistogram[i];
        offsetHistogram[i] = sum;
        sum += count;
    }
}

fn storeOffsetHistogram(globalIndex: u32) {
    let localIndex = getLocalIndex(globalIndex);
    histograms[activeHistogramRange.y][localIndex] = offsetHistogram[localIndex];
}

fn scatterKey(globalIndex: u32) {
    if (globalIndex >= keyArrayLength()) { return; }
    let workgroupIndex = getWorkgroupIndex(globalIndex);
    let localIndex = getLocalIndex(globalIndex);
    let tile = localIndex >> 4u;
    let key = localKeys[localIndex];
    let digit = getDigit(key);
    var index = countHistograms[tile][digit];

    for (var i = tile * 16u; i < localIndex; i++) {
        var otherDigit = getDigit(localKeys[i]);
        index += u32(otherDigit == digit);
    }

    index += histograms[workgroupIndex][digit];

    if (bool(shift & 4u)) {
        keysA[index] = key;
    } else {
        keysB[index] = key;
    }
}

fn storeKeyHistogram(globalIndex: u32) {
    let localIndex = getLocalIndex(globalIndex);
    let workgroupIndex = getWorkgroupIndex(globalIndex);
    if (localIndex >= 16u) { return; }

    var count = 0u;
    for (var i = 0u; i < 16u; i++) {
        count += countHistograms[i][localIndex];
    }
    histograms[workgroupIndex][localIndex] = count;
}

fn countHistogramTile(globalIndex: u32) {
    let localIndex = getLocalIndex(globalIndex);
    let workgroupIndex = getWorkgroupIndex(globalIndex);
    let lane = localIndex & 15u;
    let tile = localIndex >> 4u;

    var count = 0u;
    let globalOffset = activeHistogramRange.x;
    let workgroupOffset = globalOffset + workgroupIndex * 256u;
    let tileStart = workgroupOffset + tile * 16u;
    let tileEnd = min(tileStart + 16u, activeHistogramRange.y);

    for (var i = tileStart; i < tileEnd; i++) {
        count += histograms[i][lane];
    }

    countHistograms[tile][lane] = count;
}

fn scanHistogramTile(globalIndex: u32) {
    let localIndex = getLocalIndex(globalIndex);
    let lane = localIndex & 15u;
    let tile = localIndex >> 4u;
    var count = countHistograms[tile][lane];

    let tileStart = activeHistogramRange.x + getWorkgroupOffset(globalIndex) + tile * 16u;
    let tileEnd = min(tileStart + 16u, activeHistogramRange.y);

    for (var i = tileStart; i < tileEnd; i++) {
        let value = histograms[i][lane];
        histograms[i][lane] = count;
        count += value;
    }
}

fn storeMergedHistogram(globalIndex: u32) {
    let localIndex = getLocalIndex(globalIndex);
    let workgroupIndex = getWorkgroupIndex(globalIndex);
    if (localIndex >= 16u) { return; }

    offsetHistogram[localIndex] = 0u;
    let sum = scanLocalHistograms(globalIndex);
    let targetIndex = activeHistogramRange.y + workgroupIndex;
    histograms[targetIndex][localIndex] = sum;
}

fn countTileKeys(globalIndex: u32) {
    let workgroupOffset = getWorkgroupOffset(globalIndex);
    let numKeys = keyArrayLength() - workgroupOffset;
    let localIndex = getLocalIndex(globalIndex);
    let lane = localIndex & 15u;
    let tile = localIndex >> 4u;
    let tileOffset = tile * 16u;

    var count = 0u;
    for (var i = tileOffset; i < tileOffset + 16u; i++) {
        let digit = getDigit(localKeys[i]);
        count += u32(digit == lane && i < numKeys);
    }

    countHistograms[tile][lane] = count;
}

fn loadLocalKeys(globalIndex: u32) {
    let validA = globalIndex < arrayLength(&keysA);
    let validB = globalIndex < arrayLength(&keysB);
    let localIndex = getLocalIndex(globalIndex);
    let sourceB = bool(shift & 4u);
    var key = 0u;

    if (validA && !sourceB) {
        key = keysA[globalIndex];
    } else if (validB && sourceB) {
        key = keysB[globalIndex];
    }

    localKeys[localIndex] = key;
}

fn loadOffsetHistogram(globalIndex: u32) {
    let localIndex = getLocalIndex(globalIndex);
    let workgroupIndex = getWorkgroupIndex(globalIndex);
    if (localIndex >= 16u) { return; }
    let index = activeHistogramRange.y + workgroupIndex;
    offsetHistogram[localIndex] = histograms[index][localIndex];
}

fn keyArrayLength() -> u32 {
    if (bool(shift & 4u)) {
        return arrayLength(&keysB);
    } else {
        return arrayLength(&keysA);
    }
}

fn getDigit(key: u32) -> u32 {
    return (key >> shift) & 15u;
}

fn getLocalIndex(globalIndex: u32) -> u32 {
    return globalIndex & 255u;
}

fn getWorkgroupIndex(globalIndex: u32) -> u32 {
    return globalIndex >> 8u;
}

fn getWorkgroupOffset(globalIndex: u32) -> u32 {
    return getWorkgroupIndex(globalIndex) * 256u;
}

fn divCeil256(num: u32) -> u32 {
    return ((num - 1u) >> 8u) + 1u;
}
