// Copyright (c) 2026 UltiMaker
// libuvula is released under the terms of the LGPLv3 or higher.

#include "charting.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <queue>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include "unwrap.h"

namespace
{

constexpr uint32_t UNASSIGNED = UINT32_MAX;

struct PosKey
{
    std::array<uint32_t, 3> bits;
    bool operator==(const PosKey& o) const
    {
        return bits == o.bits;
    }
};

struct PosHash
{
    size_t operator()(const PosKey& k) const noexcept
    {
        uint64_t h = 1469598103934665603ull;
        for (const uint32_t v : k.bits)
        {
            h ^= v;
            h *= 1099511628211ull;
        }
        return static_cast<size_t>(h);
    }
};

PosKey keyOf(const Point3F& p)
{
    PosKey k{};
    const float f[3] = { p.x(), p.y(), p.z() };
    std::memcpy(k.bits.data(), f, sizeof(f));
    return k;
}

uint32_t corner(const Face& f, const int slot)
{
    return slot == 0 ? f.i1 : (slot == 1 ? f.i2 : f.i3);
}

int32_t& adjRef(FaceSigned& a, const int slot)
{
    return slot == 0 ? a.i1 : (slot == 1 ? a.i2 : a.i3);
}

int32_t adjAt(const FaceSigned& a, const int slot)
{
    return slot == 0 ? a.i1 : (slot == 1 ? a.i2 : a.i3);
}

struct EdgeRef
{
    uint64_t key;
    uint32_t face;
    uint8_t slot;
};

struct HeapItem
{
    float cost;
    uint32_t face;
    uint32_t chart;
};

struct HeapCmp
{
    bool operator()(const HeapItem& a, const HeapItem& b) const
    {
        return a.cost > b.cost; // min-heap on cost
    }
};

using Heap = std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp>;

} // namespace

MeshTopology buildTopology(const std::vector<Point3F>& vertices, const std::vector<Face>& faces)
{
    MeshTopology t;
    const size_t F = faces.size();

    // Position-welded vertex indices for adjacency
    std::unordered_map<PosKey, uint32_t, PosHash> unique;
    unique.reserve(vertices.size());
    std::vector<uint32_t> remap(vertices.size());
    for (size_t v = 0; v < vertices.size(); ++v)
    {
        const auto [it, inserted] = unique.try_emplace(keyOf(vertices[v]), static_cast<uint32_t>(v));
        remap[v] = it->second;
    }
    t.welded.resize(F);
    for (size_t f = 0; f < F; ++f)
        t.welded[f] = Face{ remap[faces[f].i1], remap[faces[f].i2], remap[faces[f].i3] };

    // Normals, areas, edge lengths
    t.normal.resize(F);
    t.area.resize(F);
    t.edge_len.resize(3 * F);
    t.degenerate.assign(F, 0);
    for (size_t f = 0; f < F; ++f)
    {
        const Point3F& a = vertices[faces[f].i1];
        const Point3F& b = vertices[faces[f].i2];
        const Point3F& c = vertices[faces[f].i3];
        Vector3F n = Vector3F(a, b).cross(Vector3F(a, c));
        const float len = n.length();
        t.area[f] = 0.5f * len;
        if (len > 1e-20f)
        {
            n /= len;
            t.normal[f] = n;
        }
        else
        {
            t.degenerate[f] = 1;
            t.normal[f] = Vector3F(0.0f, 0.0f, 1.0f);
        }
        t.edge_len[3 * f + 0] = Vector3F(a, b).length();
        t.edge_len[3 * f + 1] = Vector3F(b, c).length();
        t.edge_len[3 * f + 2] = Vector3F(c, a).length();
    }

    // Edge adjacency: sort all (edge key, face, slot) and pair up equal keys.
    std::vector<EdgeRef> refs;
    refs.reserve(3 * F);
    for (size_t f = 0; f < F; ++f)
    {
        for (int s = 0; s < 3; ++s)
        {
            uint32_t a = corner(t.welded[f], s), b = corner(t.welded[f], (s + 1) % 3);
            if (a > b)
                std::swap(a, b);
            refs.push_back(EdgeRef{ (static_cast<uint64_t>(a) << 32) | b, static_cast<uint32_t>(f), static_cast<uint8_t>(s) });
        }
    }
    std::sort(refs.begin(), refs.end(), [](const EdgeRef& x, const EdgeRef& y) { return x.key < y.key; });
    t.adj.assign(F, FaceSigned{ -1, -1, -1 });
    for (size_t i = 0; i < refs.size();)
    {
        size_t j = i + 1;
        while (j < refs.size() && refs[j].key == refs[i].key)
            ++j;
        if (j - i >= 2)
        {
            // Manifold edge: link the pair. Non-manifold fans: everyone links to the first face,
            // which keeps connectivity without creating inconsistent cycles.
            for (size_t k = i; k < j; ++k)
            {
                const size_t partner = (k == i) ? i + 1 : i;
                adjRef(t.adj[refs[k].face], refs[k].slot) = static_cast<int32_t>(refs[partner].face);
            }
        }
        i = j;
    }

    // Orientation consistency: propagate a flip flag across edges. Two consistently
    // oriented faces traverse their shared edge in opposite directions.
    std::vector<uint8_t> flip(F, 0), visited(F, 0);
    std::vector<uint32_t> stack;
    size_t flipped = 0;
    for (size_t root = 0; root < F; ++root)
    {
        if (visited[root])
            continue;
        visited[root] = 1;
        stack.clear();
        stack.push_back(static_cast<uint32_t>(root));
        while (! stack.empty())
        {
            const uint32_t f = stack.back();
            stack.pop_back();
            for (int s = 0; s < 3; ++s)
            {
                const int32_t g = adjAt(t.adj[f], s);
                if (g < 0 || visited[g])
                    continue;
                const uint32_t a = corner(t.welded[f], s), b = corner(t.welded[f], (s + 1) % 3);
                bool same_direction = false;
                for (int sg = 0; sg < 3; ++sg)
                {
                    const uint32_t ga = corner(t.welded[g], sg), gb = corner(t.welded[g], (sg + 1) % 3);
                    if (ga == a && gb == b)
                        same_direction = true;
                }
                flip[g] = static_cast<uint8_t>(flip[f] ^ (same_direction ? 1 : 0));
                visited[g] = 1;
                stack.push_back(static_cast<uint32_t>(g));
            }
        }
    }
    for (size_t f = 0; f < F; ++f)
        if (flip[f])
        {
            t.normal[f] *= -1.0f;
            ++flipped;
        }
    if (flipped > 0)
        spdlog::debug("topology: {} faces with inconsistent winding (normals flipped for charting)", flipped);
    return t;
}

namespace
{

/*!
 * Agglomerative merge of adjacent charts whose union still fits one cone: the merged
 * projection normal is the area-weighted mean and every face of both charts must lie
 * within the cone of it. Pairs are tried in order of shared boundary length. The exact
 * test runs over the smaller chart; the larger one is accepted through the angular bound
 * (its worst face angle + the normal's rotation) and tested exactly only when the bound
 * fails. Returns the number of merges.
 */
size_t mergeCharts(const MeshTopology& topo, Charting& ch, const float cos_cone, const float cone_rad, const int passes)
{
    const size_t F = topo.normal.size();
    size_t merged_total = 0;
    for (int pass = 0; pass < passes; ++pass)
    {
        const size_t C = ch.chart_normal.size();
        std::vector<std::vector<uint32_t>> members(C);
        std::vector<Vector3F> sum(C);
        for (size_t f = 0; f < F; ++f)
        {
            const uint32_t c = ch.face_chart[f];
            if (c == UNASSIGNED || topo.degenerate[f])
                continue;
            members[c].push_back(static_cast<uint32_t>(f));
            sum[c] += topo.normal[f] * topo.area[f];
        }
        std::vector<Vector3F> normal(C);
        std::vector<float> min_dot(C, 1.0f);
        for (size_t c = 0; c < C; ++c)
        {
            normal[c] = sum[c];
            if (! normal[c].normalize())
                normal[c] = ch.chart_normal[c];
            for (const uint32_t f : members[c])
                min_dot[c] = std::min(min_dot[c], topo.normal[f].dot(normal[c]));
        }
        // adjacency weighted by shared edge count
        std::unordered_map<uint64_t, uint32_t> shared;
        for (size_t f = 0; f < F; ++f)
        {
            const uint32_t a = ch.face_chart[f];
            if (a == UNASSIGNED)
                continue;
            for (int s = 0; s < 3; ++s)
            {
                const int32_t nb = adjAt(topo.adj[f], s);
                if (nb < 0)
                    continue;
                const uint32_t b = ch.face_chart[nb];
                if (b == UNASSIGNED || b <= a)
                    continue;
                ++shared[(static_cast<uint64_t>(a) << 32) | b];
            }
        }
        struct Pair
        {
            uint32_t weight, a, b;
        };
        std::vector<Pair> pairs;
        pairs.reserve(shared.size());
        for (const auto& [key, w] : shared)
            pairs.push_back(Pair{ w, static_cast<uint32_t>(key >> 32), static_cast<uint32_t>(key & 0xffffffffu) });
        std::sort(pairs.begin(), pairs.end(), [](const Pair& x, const Pair& y) { return x.weight > y.weight; });

        std::vector<uint32_t> parent(C);
        for (size_t c = 0; c < C; ++c)
            parent[c] = static_cast<uint32_t>(c);
        const auto find = [&](uint32_t x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };
        size_t merged = 0;
        for (const Pair& pr : pairs)
        {
            uint32_t ra = find(pr.a), rb = find(pr.b);
            if (ra == rb)
                continue;
            if (members[ra].size() < members[rb].size())
                std::swap(ra, rb); // ra = large, rb = small
            Vector3F n_new = sum[ra] + sum[rb];
            if (! n_new.normalize())
                continue;
            // large side: angular bound first, exact only if the bound fails
            const float rot = std::acos(std::clamp(normal[ra].dot(n_new), -1.0f, 1.0f));
            bool ok = std::acos(std::clamp(min_dot[ra], -1.0f, 1.0f)) + rot <= cone_rad;
            float new_min = 1.0f;
            if (! ok)
            {
                ok = true;
                for (const uint32_t f : members[ra])
                {
                    const float d = topo.normal[f].dot(n_new);
                    if (d < cos_cone)
                    {
                        ok = false;
                        break;
                    }
                    new_min = std::min(new_min, d);
                }
            }
            else
            {
                new_min = std::cos(std::min(std::acos(std::clamp(min_dot[ra], -1.0f, 1.0f)) + rot, cone_rad));
            }
            if (! ok)
                continue;
            for (const uint32_t f : members[rb])
            {
                const float d = topo.normal[f].dot(n_new);
                if (d < cos_cone)
                {
                    ok = false;
                    break;
                }
                new_min = std::min(new_min, d);
            }
            if (! ok)
                continue;
            parent[rb] = ra;
            sum[ra] += sum[rb];
            normal[ra] = n_new;
            min_dot[ra] = new_min;
            members[ra].insert(members[ra].end(), members[rb].begin(), members[rb].end());
            members[rb].clear();
            ++merged;
        }
        for (size_t f = 0; f < F; ++f)
            if (ch.face_chart[f] != UNASSIGNED)
                ch.face_chart[f] = find(ch.face_chart[f]);
        for (size_t c = 0; c < C; ++c)
            ch.chart_normal[c] = normal[find(static_cast<uint32_t>(c))];
        merged_total += merged;
        if (merged == 0)
            break;
    }
    return merged_total;
}

} // namespace

Charting growCharts(const MeshTopology& topo, const UnwrapOptions& o)
{
    const size_t F = topo.normal.size();
    const float cos_cone = std::cos(o.cone_half_angle_deg * std::numbers::pi_v<float> / 180.0f);
    Charting ch;
    ch.face_chart.assign(F, UNASSIGNED);

    // Grows charts from a pre-seeded heap: a face joins the chart it is reached from
    // first, provided it lies inside that chart's cone. Faces are only reached through an
    // assigned neighbour, so every chart is edge-connected by construction.
    const auto run_flood = [&](Heap& heap)
    {
        while (! heap.empty())
        {
            const HeapItem it = heap.top();
            heap.pop();
            if (ch.face_chart[it.face] != UNASSIGNED)
                continue;
            const Vector3F& N = ch.chart_normal[it.chart];
            if (topo.normal[it.face].dot(N) < cos_cone)
                continue;
            ch.face_chart[it.face] = it.chart;
            for (int s = 0; s < 3; ++s)
            {
                const int32_t nb = adjAt(topo.adj[it.face], s);
                if (nb < 0 || ch.face_chart[nb] != UNASSIGNED || topo.degenerate[nb])
                    continue;
                heap.push(HeapItem{ 1.0f - topo.normal[nb].dot(N), static_cast<uint32_t>(nb), it.chart });
            }
        }
    };

    // Seed order: large, locally flat faces first
    std::vector<uint32_t> order;
    order.reserve(F);
    {
        std::vector<float> score(F, 0.0f);
        for (size_t f = 0; f < F; ++f)
        {
            if (topo.degenerate[f])
                continue;
            float flat = 0.0f;
            int n = 0;
            for (int s = 0; s < 3; ++s)
            {
                const int32_t nb = adjAt(topo.adj[f], s);
                if (nb < 0 || topo.degenerate[nb])
                    continue;
                flat += topo.normal[f].dot(topo.normal[nb]);
                ++n;
            }
            flat = n > 0 ? flat / static_cast<float>(n) : 1.0f;
            score[f] = topo.area[f] * (0.5f + 0.5f * flat);
            order.push_back(static_cast<uint32_t>(f));
        }
        std::sort(order.begin(), order.end(), [&score](const uint32_t a, const uint32_t b) { return score[a] > score[b]; });
    }

    // Greedy pass: every unassigned face in seed order starts a chart with its own normal
    const auto grow_leftovers = [&]()
    {
        Heap heap;
        size_t started = 0;
        for (const uint32_t seed : order)
        {
            if (ch.face_chart[seed] != UNASSIGNED)
                continue;
            const auto c = static_cast<uint32_t>(ch.chart_normal.size());
            ch.chart_normal.push_back(topo.normal[seed]);
            heap.push(HeapItem{ 0.0f, seed, c });
            run_flood(heap);
            ++started;
        }
        return started;
    };

    size_t started = grow_leftovers();
    const float cone_rad = o.cone_half_angle_deg * std::numbers::pi_v<float> / 180.0f;
    spdlog::debug("charting: greedy pass -> {} charts, merged {}", started, mergeCharts(topo, ch, cos_cone, cone_rad, 3));

    for (int iter = 0; iter < o.refine_iterations; ++iter)
    {
        // Recentre: area-weighted chart normals, seed = face best aligned with it
        const size_t C = ch.chart_normal.size();
        std::vector<Vector3F> sum(C);
        std::vector<uint32_t> count(C, 0);
        for (size_t f = 0; f < F; ++f)
        {
            const uint32_t c = ch.face_chart[f];
            if (c == UNASSIGNED || topo.degenerate[f])
                continue;
            sum[c] += topo.normal[f] * topo.area[f];
            ++count[c];
        }
        for (size_t c = 0; c < C; ++c)
            if (sum[c].normalize())
                ch.chart_normal[c] = sum[c];
        std::vector<float> best(C, -2.0f);
        std::vector<uint32_t> seed(C, UNASSIGNED);
        for (size_t f = 0; f < F; ++f)
        {
            const uint32_t c = ch.face_chart[f];
            if (c == UNASSIGNED || topo.degenerate[f])
                continue;
            const float d = topo.normal[f].dot(ch.chart_normal[c]);
            if (d > best[c])
            {
                best[c] = d;
                seed[c] = static_cast<uint32_t>(f);
            }
        }

        // Simultaneous re-flood: charts compete, a face goes to the best-fitting chart that reaches
        // it. Every non-empty chart re-seeds, including tiny ones: crease faces form thin bands with
        // a normal of their own and must be allowed to grow into them instead of being orphaned.
        std::fill(ch.face_chart.begin(), ch.face_chart.end(), UNASSIGNED);
        Heap heap;
        for (size_t c = 0; c < C; ++c)
            if (seed[c] != UNASSIGNED && count[c] > 0)
                heap.push(HeapItem{ 0.0f, seed[c], static_cast<uint32_t>(c) });
        run_flood(heap);
        const size_t leftovers = grow_leftovers();

        // Boundary majority smoothing: a face whose neighbours mostly belong to another chart
        // moves there if it fits that chart's cone. Straightens jagged seams on blocky normals.
        size_t moved = 0;
        for (int pass = 0; pass < o.smoothing_passes; ++pass)
        {
            std::vector<uint32_t> next = ch.face_chart;
            for (size_t f = 0; f < F; ++f)
            {
                const uint32_t a = ch.face_chart[f];
                if (a == UNASSIGNED || topo.degenerate[f])
                    continue;
                uint32_t nb_chart[3];
                int n = 0;
                for (int s = 0; s < 3; ++s)
                {
                    const int32_t nb = adjAt(topo.adj[f], s);
                    if (nb >= 0 && ch.face_chart[nb] != UNASSIGNED && ch.face_chart[nb] != a)
                        nb_chart[n++] = ch.face_chart[nb];
                }
                if (n < 2)
                    continue;
                uint32_t b = UNASSIGNED;
                if (nb_chart[0] == nb_chart[1] || (n == 3 && nb_chart[0] == nb_chart[2]))
                    b = nb_chart[0];
                else if (n == 3 && nb_chart[1] == nb_chart[2])
                    b = nb_chart[1];
                if (b == UNASSIGNED || topo.normal[f].dot(ch.chart_normal[b]) < cos_cone)
                    continue;
                next[f] = b;
                ++moved;
            }
            ch.face_chart.swap(next);
        }
        spdlog::debug("charting: refine {} -> {} charts ({} leftover seeds, {} faces smoothed)", iter + 1, ch.chart_normal.size(), leftovers, moved);
    }

    spdlog::debug("charting: post-refine merge: {} merges", mergeCharts(topo, ch, cos_cone, cone_rad, 3));

    // Absorb tiny charts into a neighbouring chart under a relaxed cone. Crease slivers that fit
    // no neighbour at the nominal cone still project without folding as long as the angle stays
    // below 90 degrees; the extra stretch is paid on a few texels, whereas leaving them as their
    // own islands makes sub-texel specks. Targets must not be tiny themselves (no ping-pong).
    {
        const float cos_absorb = std::cos(std::min(o.absorb_half_angle_deg, 89.0f) * std::numbers::pi_v<float> / 180.0f);
        const auto tiny_limit = static_cast<uint32_t>(std::max(o.min_island_faces, 1));
        size_t absorbed = 0;
        for (int pass = 0; pass < 6; ++pass)
        {
            std::vector<uint32_t> count(ch.chart_normal.size(), 0);
            for (const uint32_t c : ch.face_chart)
                if (c != UNASSIGNED)
                    ++count[c];
            size_t moved = 0;
            for (size_t f = 0; f < F; ++f)
            {
                const uint32_t a = ch.face_chart[f];
                if (a == UNASSIGNED || topo.degenerate[f] || count[a] > tiny_limit)
                    continue;
                uint32_t best = UNASSIGNED;
                float best_dot = cos_absorb;
                for (int s = 0; s < 3; ++s)
                {
                    const int32_t nb = adjAt(topo.adj[f], s);
                    if (nb < 0)
                        continue;
                    const uint32_t b = ch.face_chart[nb];
                    if (b == UNASSIGNED || b == a || count[b] <= tiny_limit)
                        continue;
                    const float d = topo.normal[f].dot(ch.chart_normal[b]);
                    if (d >= best_dot)
                    {
                        best_dot = d;
                        best = b;
                    }
                }
                if (best != UNASSIGNED)
                {
                    ch.face_chart[f] = best;
                    ++moved;
                }
            }
            absorbed += moved;
            if (moved == 0)
                break;
        }
        spdlog::debug("charting: absorbed {} faces of tiny charts under the {:.0f} degree cone", absorbed, o.absorb_half_angle_deg);
    }

    // Degenerate faces: attach to the neighbour across the longest edge, then to any
    // assigned neighbour; isolated degenerate clusters get their own (zero-area) chart.
    for (int pass = 0; pass < 4; ++pass)
    {
        bool changed = false;
        for (size_t f = 0; f < F; ++f)
        {
            if (! topo.degenerate[f] || ch.face_chart[f] != UNASSIGNED)
                continue;
            int32_t best_nb = -1;
            float best_len = -1.0f;
            for (int s = 0; s < 3; ++s)
            {
                const int32_t nb = adjAt(topo.adj[f], s);
                if (nb >= 0 && ch.face_chart[nb] != UNASSIGNED && topo.edge_len[3 * f + s] > best_len)
                {
                    best_len = topo.edge_len[3 * f + s];
                    best_nb = nb;
                }
            }
            if (best_nb >= 0)
            {
                ch.face_chart[f] = ch.face_chart[best_nb];
                changed = true;
            }
        }
        if (! changed)
            break;
    }
    size_t isolated = 0;
    for (size_t f = 0; f < F; ++f)
    {
        if (ch.face_chart[f] != UNASSIGNED)
            continue;
        ch.face_chart[f] = static_cast<uint32_t>(ch.chart_normal.size());
        ch.chart_normal.push_back(topo.normal[f]);
        ++isolated;
    }
    if (isolated > 0)
        spdlog::debug("charting: {} isolated degenerate faces as own charts", isolated);
    return ch;
}

std::vector<std::vector<size_t>> chartLists(Charting& charting)
{
    const size_t C = charting.chart_normal.size();
    std::vector<uint32_t> count(C, 0);
    for (const uint32_t c : charting.face_chart)
        ++count[c];
    std::vector<uint32_t> new_id(C, UNASSIGNED);
    std::vector<Vector3F> normals;
    normals.reserve(C);
    for (size_t c = 0; c < C; ++c)
        if (count[c] > 0)
        {
            new_id[c] = static_cast<uint32_t>(normals.size());
            normals.push_back(charting.chart_normal[c]);
        }
    std::vector<std::vector<size_t>> lists(normals.size());
    for (size_t c = 0; c < C; ++c)
        if (new_id[c] != UNASSIGNED)
            lists[new_id[c]].reserve(count[c]);
    for (size_t f = 0; f < charting.face_chart.size(); ++f)
    {
        const uint32_t c = new_id[charting.face_chart[f]];
        charting.face_chart[f] = c;
        lists[c].push_back(f);
    }
    charting.chart_normal = std::move(normals);
    return lists;
}

void reconnectCharts(std::vector<std::vector<size_t>>& charts, std::vector<Vector3F>& chart_normals, const MeshTopology& topo)
{
    const size_t F = topo.adj.size();
    std::vector<uint32_t> face_chart(F, UNASSIGNED);
    for (size_t c = 0; c < charts.size(); ++c)
        for (const size_t f : charts[c])
            face_chart[f] = static_cast<uint32_t>(c);
    std::vector<uint8_t> visited(F, 0);
    std::vector<std::vector<size_t>> out;
    std::vector<Vector3F> out_normals;
    out.reserve(charts.size());
    out_normals.reserve(charts.size());
    std::vector<size_t> stack;
    for (size_t c = 0; c < charts.size(); ++c)
    {
        for (const size_t root : charts[c])
        {
            if (visited[root])
                continue;
            std::vector<size_t> component;
            visited[root] = 1;
            stack.clear();
            stack.push_back(root);
            while (! stack.empty())
            {
                const size_t f = stack.back();
                stack.pop_back();
                component.push_back(f);
                for (int s = 0; s < 3; ++s)
                {
                    const int32_t nb = adjAt(topo.adj[f], s);
                    if (nb >= 0 && ! visited[nb] && face_chart[nb] == c)
                    {
                        visited[nb] = 1;
                        stack.push_back(static_cast<size_t>(nb));
                    }
                }
            }
            out.push_back(std::move(component));
            out_normals.push_back(chart_normals[c]);
        }
    }
    charts = std::move(out);
    chart_normals = std::move(out_normals);
}
