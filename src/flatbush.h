// should sync
// - https://github.com/cubao/fast-crossing/blob/master/src/flatbush.h
// - https://github.com/cubao/headers/tree/main/include/cubao/flatbush.h

#pragma once

// based on https://github.com/IMQS/flatbush/blob/master/flatbush.h

#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <stdint.h>
#include <float.h>

#include <Eigen/Core>

namespace flatbush
{

inline uint32_t HilbertXYToIndex(uint32_t n, uint32_t x, uint32_t y);

template <typename TCoord> class FlatBush
{
  public:
    using LabelType = Eigen::Vector2i;
    using BoxType = Eigen::Vector4d;
    using BoxesType = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;
    using LabelsType = Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>;
    using PolylineType =
        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

    struct Box
    {
        int Index;
        TCoord MinX;
        TCoord MinY;
        TCoord MaxX;
        TCoord MaxY;
        bool PositiveUnion(const Box &b) const
        {
            return b.MaxX >= MinX && b.MinX <= MaxX && //
                   b.MaxY >= MinY && b.MinY <= MaxY;
        }
    };

    int NodeSize = 16;

    FlatBush();
    FlatBush(int N) { Reserve(N); }
    void Reserve(int size); // Calling this before calling
                            // add(),add()...finish() is an optimization
    int Add(TCoord minX, TCoord minY, TCoord maxX, TCoord maxY, //
            int label0 = -1,
            int label1 = -1); // Add an item, and return it's index
    int Add(const Eigen::Ref<const PolylineType> &polyline, int label0 = -1);
    void Finish();                        // Build the index
    void Search(TCoord minX, TCoord minY, //
                TCoord maxX, TCoord maxY,
                std::vector<int> &results) const;     // Search for items
    std::vector<int> Search(TCoord minX, TCoord minY, // Search for items
                            TCoord maxX, TCoord maxY) const;
    int Size() const { return NumItems; }

    Eigen::Map<const BoxesType> boxes() const
    {
        return Eigen::Map<const BoxesType>(&raw_Boxes[0][0], raw_Boxes.size(),
                                           4);
    }
    Eigen::Map<const LabelsType> labels() const
    {
        return Eigen::Map<const LabelsType>(&Labels[0][0], Labels.size(), 2);
    }

    Eigen::Vector4d box(int index) const { return raw_Boxes[index]; }
    Eigen::Vector2i label(int index) const { return Labels[index]; }

  private:
    std::vector<Box> Boxes;
    std::vector<BoxType, Eigen::aligned_allocator<BoxType>> raw_Boxes;
    std::vector<LabelType, Eigen::aligned_allocator<LabelType>> Labels;
    Box Bounds;
    std::vector<uint32_t> HilbertValues;
    std::vector<int> LevelBounds;
    int NumItems = 0;

    static Box InvertedBox();
    static void Sort(uint32_t *hilbertValues, Box *boxes, int left, int right);
};

template <typename TCoord> FlatBush<TCoord>::FlatBush()
{
    Bounds = InvertedBox();
}

template <typename TCoord> void FlatBush<TCoord>::Reserve(int size)
{
    int n = size;
    int numNodes = n;
    do {
        n = (n + NodeSize - 1) / NodeSize;
        numNodes += n;
    } while (n > 1);

    Boxes.reserve(numNodes);
    raw_Boxes.reserve(numNodes);
    Labels.reserve(numNodes);
}

template <typename TCoord>
int FlatBush<TCoord>::Add(TCoord x0, TCoord y0, TCoord x1, TCoord y1,
                          int label0, int label1)
{
    double minX = std::min(x0, x1);
    double maxX = std::max(x0, x1);
    double minY = std::min(y0, y1);
    double maxY = std::max(y0, y1);
    raw_Boxes.emplace_back(x0, y0, x1, y1);
    Labels.emplace_back(label0, label1);

    Bounds.MinX = std::min(Bounds.MinX, minX);
    Bounds.MinY = std::min(Bounds.MinY, minY);
    Bounds.MaxX = std::max(Bounds.MaxX, maxX);
    Bounds.MaxY = std::max(Bounds.MaxY, maxY);

    int index = Boxes.size();
    Boxes.push_back({index, minX, minY, maxX, maxY});
    return index;
}

template <typename TCoord>
int FlatBush<TCoord>::Add(const Eigen::Ref<const PolylineType> &polyline,
                          int label0)
{
    int index = Boxes.size();
    int r = polyline.rows();
    for (int i = 0; i < r - 1; ++i) {
        auto &prev = polyline.row(i);
        auto &curr = polyline.row(i + 1);
        Add(prev[0], prev[1], curr[0], curr[1], label0, i);
    }
    return index;
}

template <typename TCoord> void FlatBush<TCoord>::Finish()
{
    if (NodeSize < 2)
        NodeSize = 2;

    NumItems = Boxes.size();

    // calculate the total number of nodes in the R-tree to allocate space for
    // and the index of each tree level (used in search later)
    int n = NumItems;
    int numNodes = n;
    LevelBounds.push_back(n);
    do {
        n = (n + NodeSize - 1) / NodeSize;
        numNodes += n;
        LevelBounds.push_back(numNodes);
    } while (n > 1);

    TCoord width = Bounds.MaxX - Bounds.MinX;
    TCoord height = Bounds.MaxY - Bounds.MinY;

    HilbertValues.resize(Boxes.size());
    TCoord hilbertMax = TCoord((1 << 16) - 1);

    // map item centers into Hilbert coordinate space and calculate Hilbert
    // values
    for (int i = 0; i < Boxes.size(); i++) {
        const auto &b = Boxes[i];
        uint32_t x = uint32_t(hilbertMax *
                              ((b.MinX + b.MaxX) / 2 - Bounds.MinX) / width);
        uint32_t y = uint32_t(hilbertMax *
                              ((b.MinY + b.MaxY) / 2 - Bounds.MinY) / height);
        HilbertValues[i] = HilbertXYToIndex(16, x, y);
    }

    // sort items by their Hilbert value (for packing later)
    if (Boxes.size() != 0)
        Sort(&HilbertValues[0], &Boxes[0], 0, Boxes.size() - 1);

    // generate nodes at each tree level, bottom-up
    for (int i = 0, pos = 0; i < LevelBounds.size() - 1; i++) {
        int end = LevelBounds[i];

        // generate a parent node for each block of consecutive <nodeSize> nodes
        while (pos < end) {
            Box nodeBox = InvertedBox();
            nodeBox.Index = pos;

            // calculate bbox for the new node
            for (int j = 0; j < NodeSize && pos < end; j++) {
                const auto &box = Boxes[pos++];
                nodeBox.MinX = std::min(nodeBox.MinX, box.MinX);
                nodeBox.MinY = std::min(nodeBox.MinY, box.MinY);
                nodeBox.MaxX = std::max(nodeBox.MaxX, box.MaxX);
                nodeBox.MaxY = std::max(nodeBox.MaxY, box.MaxY);
            }

            // add the new node to the tree data
            Boxes.push_back(nodeBox);
        }
    }
}

template <typename TCoord>
std::vector<int> FlatBush<TCoord>::Search(TCoord minX, TCoord minY, //
                                          TCoord maxX, TCoord maxY) const
{
    std::vector<int> results;
    Search(minX, minY, maxX, maxY, results);
    return results;
}

template <typename TCoord>
void FlatBush<TCoord>::Search(TCoord minX, TCoord minY, //
                              TCoord maxX, TCoord maxY,
                              std::vector<int> &results) const
{
    if (LevelBounds.size() == 0) {
        // Must call Finish()
        return;
    }

    std::vector<int> queue;
    queue.push_back(Boxes.size() - 1);       // nodeIndex
    queue.push_back(LevelBounds.size() - 1); // level

    while (queue.size() != 0) {
        int nodeIndex = queue[queue.size() - 2];
        int level = queue[queue.size() - 1];
        queue.pop_back();
        queue.pop_back();

        // find the end index of the node
        int end = std::min(nodeIndex + NodeSize, LevelBounds[level]);

        // search through child nodes
        for (int pos = nodeIndex; pos < end; pos++) {
            // check if node bbox intersects with query bbox
            if (maxX < Boxes[pos].MinX || maxY < Boxes[pos].MinY ||
                minX > Boxes[pos].MaxX || minY > Boxes[pos].MaxY) {
                continue;
            }
            if (nodeIndex < NumItems) {
                // leaf item
                results.push_back(Boxes[pos].Index);
            } else {
                // node; add it to the search queue
                queue.push_back(Boxes[pos].Index);
                queue.push_back(level - 1);
            }
        }
    }
}

template <typename TCoord>
typename FlatBush<TCoord>::Box FlatBush<TCoord>::InvertedBox()
{
    FlatBush<TCoord>::Box b;
    b.Index = -1;
    b.MinX = std::numeric_limits<TCoord>::max();
    b.MinY = std::numeric_limits<TCoord>::max();
    b.MaxX = std::numeric_limits<TCoord>::lowest();
    b.MaxY = std::numeric_limits<TCoord>::lowest();
    return b;
}

// custom quicksort that sorts bbox data alongside the hilbert values
template <typename TCoord>
void FlatBush<TCoord>::Sort(uint32_t *values, Box *boxes, int left, int right)
{
    if (left >= right)
        return;

    uint32_t pivot = values[(left + right) >> 1];
    int i = left - 1;
    int j = right + 1;

    while (true) {
        do
            i++;
        while (values[i] < pivot);
        do
            j--;
        while (values[j] > pivot);
        if (i >= j)
            break;
        std::swap(values[i], values[j]);
        std::swap(boxes[i], boxes[j]);
    }

    Sort(values, boxes, left, j);
    Sort(values, boxes, j + 1, right);
}

// From https://github.com/rawrunprotected/hilbert_curves (public domain)
inline uint32_t Interleave(uint32_t x)
{
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

inline uint32_t HilbertXYToIndex(uint32_t n, uint32_t x, uint32_t y)
{
    x = x << (16 - n);
    y = y << (16 - n);

    uint32_t A, B, C, D;

    // Initial prefix scan round, prime with x and y
    {
        uint32_t a = x ^ y;
        uint32_t b = 0xFFFF ^ a;
        uint32_t c = 0xFFFF ^ (x | y);
        uint32_t d = x & (y ^ 0xFFFF);

        A = a | (b >> 1);
        B = (a >> 1) ^ a;

        C = ((c >> 1) ^ (b & (d >> 1))) ^ c;
        D = ((a & (c >> 1)) ^ (d >> 1)) ^ d;
    }

    {
        uint32_t a = A;
        uint32_t b = B;
        uint32_t c = C;
        uint32_t d = D;

        A = ((a & (a >> 2)) ^ (b & (b >> 2)));
        B = ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2)));

        C ^= ((a & (c >> 2)) ^ (b & (d >> 2)));
        D ^= ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2)));
    }

    {
        uint32_t a = A;
        uint32_t b = B;
        uint32_t c = C;
        uint32_t d = D;

        A = ((a & (a >> 4)) ^ (b & (b >> 4)));
        B = ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4)));

        C ^= ((a & (c >> 4)) ^ (b & (d >> 4)));
        D ^= ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4)));
    }

    // Final round and projection
    {
        uint32_t a = A;
        uint32_t b = B;
        uint32_t c = C;
        uint32_t d = D;

        C ^= ((a & (c >> 8)) ^ (b & (d >> 8)));
        D ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)));
    }

    // Undo transformation prefix scan
    uint32_t a = C ^ (C >> 1);
    uint32_t b = D ^ (D >> 1);

    // Recover index bits
    uint32_t i0 = x ^ y;
    uint32_t i1 = b | (0xFFFF ^ (i0 | a));

    return ((Interleave(i1) << 1) | Interleave(i0)) >> (32 - 2 * n);
}
} // namespace flatbush
