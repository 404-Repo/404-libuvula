// (c) 2025, UltiMaker -- see LICENCE for details

#include "unwrap.h"

#include "charting.h"

#include <chrono>

#include <algorithm>
#include <atomic>
#include <future>
#include <cmath>
#include <map>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>

#include <range/v3/algorithm/partition.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/map.hpp>
#include <spdlog/spdlog.h>

#include "Matrix33F.h"
#include "Point2F.h"
#include "Point3F.h"
#include "Vector3F.h"
#include "geometry_utils.h"
#include "xatlas.h"


struct FaceData
{
    const Face* face;
    size_t face_index;
    Vector3F normal;
};

/*!
 * Calculate the best projection normals according to the given input faces
 * @param faces_data The faces data
 * @return A list of normals that are far enough from each other
 */
std::vector<Vector3F> calculateProjectionNormals(const std::vector<FaceData>& faces_data)
{
    constexpr float group_angle_limit = 20.0;

    const float group_angle_limit_cos = std::cos(geometry_utils::deg2rad(group_angle_limit));
    const float group_angle_limit_half_cos = std::cos(geometry_utils::deg2rad(group_angle_limit / 2));

    // First group will be based on the normal of the very first face
    const Vector3F* project_normal = &faces_data.front().normal;

    std::vector<Vector3F> projection_normals;
    std::vector<float> best_angles(faces_data.size(), std::numeric_limits<float>::lowest());
    size_t previous_normal_count = 0;

    // Create an internal list containing pointers to all the faces data, it will be reorganized
    std::vector<const FaceData*> faces_to_process(faces_data.size());
    std::transform(
        faces_data.begin(),
        faces_data.end(),
        faces_to_process.begin(),
        [](const FaceData& face_data)
        {
            return &face_data;
        });

    using FaceDataIterator = std::vector<const FaceData*>::iterator;
    struct FaceDataRange
    {
        FaceDataIterator begin;
        FaceDataIterator end;
    };

    // The unprocessed_faces is a sub-range of the faces list, that contains all the faces that have not been assigned to a group yet.
    FaceDataRange unprocessed_faces = { .begin = std::next(faces_to_process.begin()), .end = faces_to_process.end() };

    while (true)
    {
        // Get all the faces that belong to the group of the current projection normal,
        // by placing them at the beginning of the unprocessed faces
        FaceDataRange current_faces_group{ .begin = unprocessed_faces.begin };
        current_faces_group.end = ranges::partition(
            unprocessed_faces.begin,
            unprocessed_faces.end,
            [&project_normal, &group_angle_limit_half_cos](const FaceData* face_data)
            {
                return face_data->normal.dot(*project_normal) > group_angle_limit_half_cos;
            });

        // All the faces placed to the current group are now no more in the unprocessed faces
        unprocessed_faces.begin = current_faces_group.end;

        // Sum all the normals of the current faces group to get the average direction
        Vector3F summed_normals = std::accumulate(
            current_faces_group.begin,
            current_faces_group.end,
            *project_normal,
            [](const Vector3F& normal, const FaceData* face_data)
            {
                return normal + face_data->normal;
            });
        if (summed_normals.normalize()) [[likely]]
        {
            projection_normals.push_back(summed_normals);
        }

        // For the next iteration, try to find the most different remaining normal from all generated normals
        float best_outlier_angle = std::numeric_limits<float>::max();
        FaceDataIterator best_outlier_face = faces_to_process.end();

        for (auto iterator = unprocessed_faces.begin; iterator != unprocessed_faces.end; ++iterator)
        {
            // Existing normals and face normals are immutable; extend their
            // previous maximum in exactly the original comparison order.
            float& face_best_angle = best_angles[*iterator - faces_data.data()];
            for (size_t i = previous_normal_count; i < projection_normals.size(); ++i)
            {
                face_best_angle = std::max(face_best_angle, projection_normals[i].dot((*iterator)->normal));
            }

            if (face_best_angle < best_outlier_angle)
            {
                best_outlier_angle = face_best_angle;
                best_outlier_face = iterator;
            }
        }

        previous_normal_count = projection_normals.size();

        if (best_outlier_angle < group_angle_limit_cos)
        {
            // Take the normal of the best outlier as the base for the iteration of the next group
            project_normal = &(*best_outlier_face)->normal;

            // Remove the faces from the unprocessed faces
            std::iter_swap(best_outlier_face, unprocessed_faces.begin);
            ++unprocessed_faces.begin;
        }
        else if (! projection_normals.empty())
        {
            break;
        }
    }

    return projection_normals;
}

static std::vector<FaceData> makeFacesData(const std::vector<Point3F>& vertices, const std::vector<Face>& faces)
{
    std::vector<FaceData> faces_data;
    faces_data.reserve(faces.size());

    for (const auto& [index, face] : faces | ranges::views::enumerate)
    {
        const Point3F& v1 = vertices[face.i1];
        const Point3F& v2 = vertices[face.i2];
        const Point3F& v3 = vertices[face.i3];

        const std::optional<Vector3F> triangle_normal = geometry_utils::triangleNormal(v1, v2, v3);
        if (triangle_normal.has_value())
        {
            faces_data.push_back(FaceData{ &face, index, triangle_normal.value() });
        }
    }

    return faces_data;
}

/*!
 * Groups the faces that have a similar normal, and project their points as raw UV coordinates along this normal
 * @param vertices The list of vertices positions
 * @param faces The list of faces we want to project
 * @param uv_coords The UV coordinates, which should be properly sized but the input content doesn't matter. As output, they will be filled with
 *                  raw UV coordinates that overlap and are not in the [0,1] range
 * @return A list containing grouped indices of faces
 */
static std::vector<std::vector<size_t>> makeCharts(const std::vector<Point3F>& vertices, const std::vector<Face>& faces, std::vector<Point2F>& uv_coords)
{
    const std::vector<FaceData> faces_data = makeFacesData(vertices, faces);
    if (faces_data.empty()) [[unlikely]]
    {
        return {};
    }

    // Calculate the best normals to group the faces
    const std::vector<Vector3F> project_normal_array = calculateProjectionNormals(faces_data);
    if (project_normal_array.empty()) [[unlikely]]
    {
        return {};
    }

    // For each face, find the best projection normal and make groups
    std::map<const Vector3F*, std::vector<const FaceData*>> projected_faces_groups;
    for (const FaceData& face_data : faces_data)
    {
        const Vector3F* best_projection_normal = nullptr;
        float angle_best = std::numeric_limits<float>::lowest();

        for (const Vector3F& projection_normal : project_normal_array)
        {
            const float angle = face_data.normal.dot(projection_normal);
            if (angle > angle_best)
            {
                angle_best = angle;
                best_projection_normal = &projection_normal;
            }
        }

        projected_faces_groups[best_projection_normal].push_back(&face_data);
    }

    // Now project each faces according to the closest matching normal and create indices groups
    std::vector<std::vector<size_t>> grouped_faces_indices;
    for (const auto& [normal, faces_data] : projected_faces_groups)
    {
        const Matrix33F axis_mat = Matrix33F::makeOrthogonalBasis(*normal);
        std::vector<size_t> faces_group;
        faces_group.reserve(faces_data.size());

        for (const FaceData* face_from_group : faces_data)
        {
            faces_group.push_back(face_from_group->face_index);

            for (const uint32_t vertex_index : { face_from_group->face->i1, face_from_group->face->i2, face_from_group->face->i3 })
            {
                uv_coords[vertex_index] = axis_mat.project(vertices[vertex_index]);
            }
        }

        grouped_faces_indices.push_back(std::move(faces_group));
    }

    return grouped_faces_indices;
}

/*!
 * Splits each normal-group into connected components using edge adjacency (BFS).
 * Two faces are considered connected only if they share a full edge (two vertices),
 * not merely a single vertex.  This avoids the problem where the old vertex-based
 * connectivity merged distant faces that touched only at a corner.
 * @param grouped_faces Contains the grouped indices of faces from makeCharts
 * @param faces Position-merged faces for adjacency detection (@sa groupSimilarVertices)
 * @return Grouped faces with groups containing only edge-connected faces
 */
std::vector<std::vector<size_t>> splitNonLinkedFacesCharts(const std::vector<std::vector<size_t>>& grouped_faces, const std::vector<Face>& faces)
{
    std::vector<std::vector<size_t>> result;

    for (const std::vector<size_t>& faces_group : grouped_faces)
    {
        if (faces_group.size() <= 1)
        {
            result.push_back(faces_group);
            continue;
        }

        // Connectivity alone determines each component. Union edge neighbors
        // without constructing every pair in an adjacency list. Choosing the
        // smallest local root preserves the original BFS seed/component order.
        std::vector<size_t> parent(faces_group.size());
        std::iota(parent.begin(), parent.end(), 0);
        const auto find_root = [&](size_t i) {
            while (parent[i] != i)
            {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        std::unordered_map<uint64_t, size_t> edge_first;
        edge_first.reserve(faces_group.size() * 2);
        for (size_t li = 0; li < faces_group.size(); ++li)
        {
            const Face& face = faces[faces_group[li]];
            const uint32_t v[3] = { face.i1, face.i2, face.i3 };
            for (int e = 0; e < 3; ++e)
            {
                const uint32_t a = std::min(v[e], v[(e + 1) % 3]);
                const uint32_t b = std::max(v[e], v[(e + 1) % 3]);
                const uint64_t key = (uint64_t(a) << 32) | uint64_t(b);
                const auto [entry, inserted] = edge_first.emplace(key, li);
                if (!inserted)
                {
                    const size_t x = find_root(li), y = find_root(entry->second);
                    parent[std::max(x, y)] = std::min(x, y);
                }
            }
        }
        std::map<size_t, std::vector<size_t>> components;
        for (size_t li = 0; li < faces_group.size(); ++li)
            components[find_root(li)].push_back(faces_group[li]);
        for (auto& [root, component] : components)
            result.push_back(std::move(component));
    }

    return result;
}

/*!
 * When loading the mesh, each vertex of each triangle is given a unique index, even if it is used in multiple adjacent triangles. The purpose
 * of this function is to remove double vertices so that we can make adjacency detection easier.
 * @param faces The original list of faces
 * @param vertices The original list of vertices position
 * @return The modified list of faces, which contains as many faces but with merged vertices
 */
std::vector<Face> groupSimilarVertices(const std::vector<Face>& faces, const std::vector<Point3F>& vertices)
{
    std::vector<Face> faces_with_similar_indices;
    std::map<Point3F, size_t> unique_vertices_indices;
    std::vector<uint32_t> new_vertices_indices(vertices.size());

    for (const auto [index, vertex] : vertices | ranges::views::enumerate)
    {
        auto iterator = unique_vertices_indices.find(vertex);
        if (iterator == unique_vertices_indices.end())
        {
            // This is the very first time we see this position, register it
            unique_vertices_indices[vertex] = index;
            new_vertices_indices[index] = index;
        }
        else
        {
            new_vertices_indices[index] = iterator->second;
        }
    }

    for (const Face& face : faces)
    {
        faces_with_similar_indices.push_back(Face{ new_vertices_indices[face.i1], new_vertices_indices[face.i2], new_vertices_indices[face.i3] });
    }

    return faces_with_similar_indices;
}

/*!
 * Detects self-overlapping UV projections within charts and splits them into non-overlapping layers.
 * When a curved surface is projected orthographically, the projection can fold over itself. This function
 * rasterizes each chart's UV triangles onto a grid (barycentric point-in-triangle) to detect such overlap,
 * then uses BFS-based layering to partition each overlapping chart into non-overlapping sub-charts.
 */
static std::vector<std::vector<size_t>> splitSelfOverlappingCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& faces,
    const std::vector<Point2F>& uv_coords)
{
    constexpr int GRID_RES = 128;
    constexpr int GRID_CELLS = GRID_RES * GRID_RES;
    std::vector<std::vector<size_t>> result;

    std::vector<std::vector<std::vector<size_t>>> per_chart(charts.size());
    auto process_chart = [&](size_t chart_index)
    {
        const auto& chart = charts[chart_index];
        auto& result = per_chart[chart_index];
        if (chart.size() <= 2)
        {
            result.push_back(chart);
            return;
        }

        float umin = std::numeric_limits<float>::max(), umax = std::numeric_limits<float>::lowest();
        float vmin = std::numeric_limits<float>::max(), vmax = std::numeric_limits<float>::lowest();
        for (const size_t fi : chart)
        {
            for (const uint32_t vi : { faces[fi].i1, faces[fi].i2, faces[fi].i3 })
            {
                umin = std::min(umin, uv_coords[vi].x);
                umax = std::max(umax, uv_coords[vi].x);
                vmin = std::min(vmin, uv_coords[vi].y);
                vmax = std::max(vmax, uv_coords[vi].y);
            }
        }

        const float du = umax - umin;
        const float dv = vmax - vmin;
        if (du < 1e-10f || dv < 1e-10f)
        {
            result.push_back(chart);
            return;
        }

        const float scale_u = static_cast<float>(GRID_RES) / du;
        const float scale_v = static_cast<float>(GRID_RES) / dv;

        // Build face adjacency within this chart (faces sharing a vertex are adjacent)
        std::map<uint32_t, std::vector<size_t>> vert_faces;
        for (size_t i = 0; i < chart.size(); i++)
        {
            const Face& f = faces[chart[i]];
            vert_faces[f.i1].push_back(i);
            vert_faces[f.i2].push_back(i);
            vert_faces[f.i3].push_back(i);
        }

        std::vector<std::vector<size_t>> adj(chart.size());
        for (const auto& [v, flist] : vert_faces)
            for (size_t a : flist)
                for (size_t b : flist)
                    if (a != b)
                        adj[a].push_back(b);
        for (auto& a : adj)
        {
            std::sort(a.begin(), a.end());
            a.erase(std::unique(a.begin(), a.end()), a.end());
        }

        // Rasterize a face's UV triangle onto the grid using barycentric point-in-triangle
        auto rasterize = [&](size_t local_idx, std::vector<int>& out)
        {
            out.clear();
            const Face& face = faces[chart[local_idx]];
            const float gx0 = (uv_coords[face.i1].x - umin) * scale_u;
            const float gy0 = (uv_coords[face.i1].y - vmin) * scale_v;
            const float gx1 = (uv_coords[face.i2].x - umin) * scale_u;
            const float gy1 = (uv_coords[face.i2].y - vmin) * scale_v;
            const float gx2 = (uv_coords[face.i3].x - umin) * scale_u;
            const float gy2 = (uv_coords[face.i3].y - vmin) * scale_v;

            const int x0 = std::max(0, static_cast<int>(std::min({ gx0, gx1, gx2 })));
            const int x1 = std::min(GRID_RES - 1, static_cast<int>(std::max({ gx0, gx1, gx2 })));
            const int y0 = std::max(0, static_cast<int>(std::min({ gy0, gy1, gy2 })));
            const int y1 = std::min(GRID_RES - 1, static_cast<int>(std::max({ gy0, gy1, gy2 })));

            const float det = (gx1 - gx0) * (gy2 - gy0) - (gx2 - gx0) * (gy1 - gy0);
            if (std::abs(det) < 1e-10f)
                return;
            const float inv_det = 1.0f / det;

            for (int gy = y0; gy <= y1; gy++)
            {
                const float py = gy + 0.5f;
                for (int gx = x0; gx <= x1; gx++)
                {
                    const float px = gx + 0.5f;
                    const float ba = ((gx1 - px) * (gy2 - py) - (gx2 - px) * (gy1 - py)) * inv_det;
                    const float bb = ((gx2 - px) * (gy0 - py) - (gx0 - px) * (gy2 - py)) * inv_det;
                    const float bc = 1.0f - ba - bb;
                    if (ba > 0.0f && bb > 0.0f && bc > 0.0f)
                        out.push_back(gy * GRID_RES + gx);
                }
            }
        };

        // Quick check: do any non-adjacent faces overlap in UV space?
        std::vector<size_t> detect_grid(GRID_CELLS, SIZE_MAX);
        bool overlap_found = false;
        std::vector<int> cells;

        for (size_t i = 0; i < chart.size() && ! overlap_found; i++)
        {
            rasterize(i, cells);
            for (const int c : cells)
            {
                if (detect_grid[c] != SIZE_MAX && detect_grid[c] != i)
                {
                    if (! std::binary_search(adj[i].begin(), adj[i].end(), detect_grid[c]))
                    {
                        overlap_found = true;
                        break;
                    }
                }
                detect_grid[c] = i;
            }
        }

        if (! overlap_found)
        {
            result.push_back(chart);
            return;
        }

        // BFS layering: greedily assign faces to non-overlapping layers while preserving connectivity.
        // Each layer gets a grid tracking all occupying faces per cell so that adjacency checks
        // correctly distinguish edge-sharing (harmless) from true overlap.
        std::vector<int> face_layer(chart.size(), -1);
        int layer_count = 0;
        std::vector<std::vector<size_t>> lgrid(GRID_CELLS);

        while (true)
        {
            size_t seed = SIZE_MAX;
            for (size_t i = 0; i < chart.size(); i++)
                if (face_layer[i] == -1)
                {
                    seed = i;
                    break;
                }
            if (seed == SIZE_MAX)
                break;

            const int layer = layer_count++;
            for (auto& cell : lgrid)
                cell.clear();

            std::vector<size_t> queue;
            size_t front = 0;
            queue.push_back(seed);

            while (front < queue.size())
            {
                const size_t curr = queue[front++];
                if (face_layer[curr] != -1)
                    continue;

                rasterize(curr, cells);

                bool conflict = false;
                for (const int c : cells)
                {
                    for (const size_t occ : lgrid[c])
                    {
                        if (occ != curr && ! std::binary_search(adj[curr].begin(), adj[curr].end(), occ))
                        {
                            conflict = true;
                            break;
                        }
                    }
                    if (conflict)
                        break;
                }

                if (conflict)
                    continue;

                face_layer[curr] = layer;
                for (const int c : cells)
                    lgrid[c].push_back(curr);

                for (const size_t neighbor : adj[curr])
                    if (face_layer[neighbor] == -1)
                        queue.push_back(neighbor);
            }
        }

        std::vector<std::vector<size_t>> layers(layer_count);
        for (size_t i = 0; i < chart.size(); i++)
            if (face_layer[i] >= 0)
                layers[face_layer[i]].push_back(chart[i]);

        for (auto& layer : layers)
            if (! layer.empty())
                result.push_back(std::move(layer));
    };

    // Charts have disjoint local state. Retain their original output ordering.
    const size_t worker_count = std::min<size_t>(16, charts.size());
    std::atomic<size_t> next_chart{0};
    std::vector<std::future<void>> workers;
    for (size_t worker = 0; worker < worker_count; ++worker)
        workers.push_back(std::async(std::launch::async, [&] {
            while (true) {
                const size_t index = next_chart.fetch_add(1, std::memory_order_relaxed);
                if (index >= charts.size()) return;
                process_chart(index);
            }
        }));
    for (auto& worker : workers) worker.get();
    for (auto& group : per_chart)
        for (auto& chart : group)
            result.push_back(std::move(chart));

    return result;
}

/*!
 * Splits charts that contain a UV fold — faces whose UV triangles have opposite winding
 * from the chart majority.  On a curved surface projected along a single normal, the fold
 * happens where the surface crosses 90° from that normal; faces beyond the fold get a
 * flipped (negative signed-area) UV triangle.  The grid-based overlap detection misses
 * these because the folded faces are topologically adjacent (share vertices along the fold).
 */
static std::vector<std::vector<size_t>> splitFoldedCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& faces,
    const std::vector<Point2F>& uv_coords)
{
    std::vector<std::vector<size_t>> result;

    for (const auto& chart : charts)
    {
        if (chart.size() <= 2)
        {
            result.push_back(chart);
            continue;
        }

        std::vector<size_t> positive, negative;
        for (const size_t fi : chart)
        {
            const Face& f = faces[fi];
            const Point2F& a = uv_coords[f.i1];
            const Point2F& b = uv_coords[f.i2];
            const Point2F& c = uv_coords[f.i3];

            const float signed_area = (b.x - a.x) * (c.y - a.y)
                                    - (c.x - a.x) * (b.y - a.y);

            if (signed_area >= 0.0f)
                positive.push_back(fi);
            else
                negative.push_back(fi);
        }

        if (negative.empty() || positive.empty())
            result.push_back(chart);
        else
        {
            result.push_back(std::move(positive));
            result.push_back(std::move(negative));
        }
    }

    return result;
}

/*!
 * Splits charts whose UV bounding box has an extreme aspect ratio into shorter segments.
 * Long thin islands (from cylindrical surfaces, pipes, flanges) pack poorly into a square texture.
 * Splitting them into roughly square segments improves packing efficiency.
 */
static std::vector<std::vector<size_t>> splitElongatedCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& faces,
    const std::vector<Point2F>& uv_coords)
{
    constexpr float max_aspect = 8.0f;
    constexpr float target_aspect = 3.0f;
    std::vector<std::vector<size_t>> result;

    for (const auto& chart : charts)
    {
        if (chart.size() <= 4)
        {
            result.push_back(chart);
            continue;
        }

        float umin = std::numeric_limits<float>::max(), umax = std::numeric_limits<float>::lowest();
        float vmin = std::numeric_limits<float>::max(), vmax = std::numeric_limits<float>::lowest();
        for (const size_t fi : chart)
        {
            for (const uint32_t vi : { faces[fi].i1, faces[fi].i2, faces[fi].i3 })
            {
                umin = std::min(umin, uv_coords[vi].x);
                umax = std::max(umax, uv_coords[vi].x);
                vmin = std::min(vmin, uv_coords[vi].y);
                vmax = std::max(vmax, uv_coords[vi].y);
            }
        }

        const float du = umax - umin;
        const float dv = vmax - vmin;
        if (du < 1e-10f || dv < 1e-10f)
        {
            result.push_back(chart);
            continue;
        }

        const float aspect = std::max(du / dv, dv / du);
        if (aspect <= max_aspect)
        {
            result.push_back(chart);
            continue;
        }

        const bool split_along_u = (du > dv);
        const int n_segments = static_cast<int>(std::ceil(aspect / target_aspect));
        const float seg_min = split_along_u ? umin : vmin;
        const float seg_size = (split_along_u ? du : dv) / static_cast<float>(n_segments);

        std::vector<std::vector<size_t>> segments(n_segments);
        for (const size_t fi : chart)
        {
            float centroid = 0.0f;
            for (const uint32_t vi : { faces[fi].i1, faces[fi].i2, faces[fi].i3 })
                centroid += split_along_u ? uv_coords[vi].x : uv_coords[vi].y;
            centroid /= 3.0f;

            const int seg = std::clamp(static_cast<int>((centroid - seg_min) / seg_size), 0, n_segments - 1);
            segments[seg].push_back(fi);
        }

        for (auto& seg : segments)
            if (! seg.empty())
                result.push_back(std::move(seg));
    }

    return result;
}

/*!
 * Merges charts below a minimum face count into the neighboring chart they share the most boundary
 * vertices with, provided their average normals are compatible. Uses union-find for clean cycle handling.
 * @param charts Current face groups
 * @param connectivity_faces Faces with position-merged vertex indices for adjacency detection
 * @param vertex_faces Face adjacency for the unchanged connectivity indices
 * @param original_faces Original faces for normal computation
 * @param vertices Vertex positions for normal computation
 * @param min_chart_size Charts with fewer faces than this are candidates for merging
 */
static std::vector<std::vector<size_t>> mergeTinyCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& connectivity_faces,
    const std::vector<std::vector<size_t>>& vertex_faces,
    const std::vector<Face>& original_faces,
    const std::vector<Point3F>& vertices,
    size_t min_chart_size)
{
    if (charts.size() <= 1)
        return charts;

    constexpr float min_normal_alignment = 0.707f; // cos(45°)
    std::vector<Vector3F> chart_normals(charts.size());
    for (size_t ci = 0; ci < charts.size(); ci++)
    {
        Vector3F normal;
        for (const size_t fi : charts[ci])
        {
            const Face& face = original_faces[fi];
            const auto tri_normal = geometry_utils::triangleNormal(vertices[face.i1], vertices[face.i2], vertices[face.i3]);
            if (tri_normal.has_value())
                normal += tri_normal.value();
        }
        normal.normalize();
        chart_normals[ci] = normal;
    }

    std::vector<size_t> face_chart(connectivity_faces.size(), SIZE_MAX);
    for (size_t ci = 0; ci < charts.size(); ci++)
        for (const size_t fi : charts[ci])
            face_chart[fi] = ci;

    std::vector<size_t> parent(charts.size());
    std::iota(parent.begin(), parent.end(), size_t(0));
    std::vector<size_t> group_size(charts.size());
    for (size_t ci = 0; ci < charts.size(); ci++)
        group_size[ci] = charts[ci].size();

    auto find = [&](size_t x) -> size_t
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](size_t a, size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        if (group_size[a] < group_size[b])
            std::swap(a, b);
        parent[b] = a;
        group_size[a] += group_size[b];
    };

    // Reuse dense counts instead of allocating a tree for each tiny chart.
    // Sort only touched roots to retain the original map's merge tie-breaking.
    std::vector<size_t> neighbor_counts(charts.size(), 0);
    std::vector<size_t> neighbors;
    for (size_t ci = 0; ci < charts.size(); ci++)
    {
        if (charts[ci].size() >= min_chart_size)
            continue;

        neighbors.clear();
        for (const size_t fi : charts[ci])
        {
            for (const uint32_t vi : { connectivity_faces[fi].i1, connectivity_faces[fi].i2, connectivity_faces[fi].i3 })
            {
                for (const size_t nfi : vertex_faces[vi])
                {
                    const size_t nci = face_chart[nfi];
                    if (nci != SIZE_MAX && find(nci) != find(ci))
                    {
                        const size_t root = find(nci);
                        if (neighbor_counts[root]++ == 0)
                            neighbors.push_back(root);
                    }
                }
            }
        }

        if (neighbors.empty())
            continue;
        std::sort(neighbors.begin(), neighbors.end());

        size_t best_neighbor = find(ci);
        size_t best_count = 0;
        for (const size_t nci : neighbors)
        {
            const size_t count = neighbor_counts[nci];
            neighbor_counts[nci] = 0;
            if (chart_normals[ci].dot(chart_normals[nci]) < min_normal_alignment)
                continue;

            if (count > best_count)
            {
                best_count = count;
                best_neighbor = nci;
            }
        }

        if (best_neighbor != find(ci))
            unite(ci, best_neighbor);
    }

    std::map<size_t, std::vector<size_t>> merged;
    for (size_t ci = 0; ci < charts.size(); ci++)
    {
        const size_t root = find(ci);
        for (const size_t fi : charts[ci])
            merged[root].push_back(fi);
    }

    std::vector<std::vector<size_t>> result;
    result.reserve(merged.size());
    for (auto& [root, chart_faces] : merged)
        result.push_back(std::move(chart_faces));

    return result;
}

/*!
 * Duplicates vertices at chart boundaries so each chart exclusively owns its vertices,
 * then re-projects each chart's vertices along the chart's average normal.
 */
static void duplicateAndReproject(
    std::vector<Point3F>& vertices,
    std::vector<Face>& faces,
    std::vector<Point2F>& uv_coords,
    const std::vector<Face>& original_faces,
    const std::vector<std::vector<size_t>>& charts,
    std::vector<uint32_t>* out_vertex_remap = nullptr,
    const std::vector<Vector3F>* chart_normals = nullptr)
{
    std::vector<uint32_t> vertex_owner(vertices.size(), UINT32_MAX);
    std::vector<uint32_t> vertex_remap(vertices.size());
    std::iota(vertex_remap.begin(), vertex_remap.end(), 0u);
    // Charts are processed in order, so only the current chart's duplicate for
    // each source vertex can be requested again. Keep the original creation order.
    std::vector<uint32_t> duplicate_chart(vertices.size(), UINT32_MAX);
    std::vector<uint32_t> duplicate_index(vertices.size(), UINT32_MAX);

    std::vector<bool> face_in_chart(faces.size(), false);

    for (uint32_t chart_idx = 0; chart_idx < static_cast<uint32_t>(charts.size()); ++chart_idx)
    {
        for (const size_t face_idx : charts[chart_idx])
        {
            face_in_chart[face_idx] = true;
            Face& face = faces[face_idx];
            for (uint32_t* vi : { &face.i1, &face.i2, &face.i3 })
            {
                if (vertex_owner[*vi] == chart_idx)
                    continue;

                if (vertex_owner[*vi] == UINT32_MAX)
                {
                    vertex_owner[*vi] = chart_idx;
                }
                else
                {
                    if (duplicate_chart[*vi] == chart_idx)
                    {
                        *vi = duplicate_index[*vi];
                    }
                    else
                    {
                        const auto new_idx = static_cast<uint32_t>(uv_coords.size());
                        uv_coords.push_back(uv_coords[*vi]);
                        vertices.push_back(vertices[*vi]);
                        vertex_owner.push_back(chart_idx);
                        vertex_remap.push_back(*vi);
                        duplicate_chart[*vi] = chart_idx;
                        duplicate_index[*vi] = new_idx;
                        duplicate_chart.push_back(UINT32_MAX);
                        duplicate_index.push_back(UINT32_MAX);
                        *vi = new_idx;
                    }
                }
            }
        }
    }

    // Detach vertices of uncharted faces (degenerate triangles) from charted ones.
    // Without this, degenerate faces share vertices with charted faces, creating
    // long thin triangles spanning between packed UV islands or from the UV origin.
    for (size_t fi = 0; fi < faces.size(); fi++)
    {
        if (face_in_chart[fi])
            continue;
        Face& face = faces[fi];
        for (uint32_t* vi : { &face.i1, &face.i2, &face.i3 })
        {
            if (vertex_owner[*vi] != UINT32_MAX)
            {
                const auto new_idx = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertices[*vi]);
                uv_coords.push_back(Point2F{ .x = 0.0f, .y = 0.0f });
                vertex_owner.push_back(UINT32_MAX);
                vertex_remap.push_back(vertex_remap[*vi]);
                *vi = new_idx;
            }
        }
    }

    for (uint32_t chart_idx = 0; chart_idx < static_cast<uint32_t>(charts.size()); ++chart_idx)
    {
        Vector3F avg_normal;
        if (chart_normals)
        {
            // Projection direction chosen by the charter (every face is inside its cone)
            avg_normal = (*chart_normals)[chart_idx];
        }
        else
        {
            for (const size_t face_idx : charts[chart_idx])
            {
                const Face& orig_face = original_faces[face_idx];
                const auto tri_normal = geometry_utils::triangleNormal(
                    vertices[orig_face.i1], vertices[orig_face.i2], vertices[orig_face.i3]);
                if (tri_normal.has_value())
                    avg_normal += tri_normal.value();
            }
            if (! avg_normal.normalize())
                continue;
        }

        const Matrix33F axis_mat = Matrix33F::makeOrthogonalBasis(avg_normal);

        for (const size_t face_idx : charts[chart_idx])
        {
            const Face& face = faces[face_idx];
            for (const uint32_t vi : { face.i1, face.i2, face.i3 })
            {
                uv_coords[vi] = axis_mat.project(vertices[vertex_remap[vi]]);
            }
        }

    }

    if (out_vertex_remap)
        *out_vertex_remap = std::move(vertex_remap);
}

static void restorePreDuplicationState(
    std::vector<Point3F>& vertices,
    std::vector<Point2F>& uv_coords,
    std::vector<Face>& faces,
    const std::vector<Face>& original_faces,
    size_t original_vertex_count)
{
    vertices.erase(vertices.begin() + static_cast<std::ptrdiff_t>(original_vertex_count), vertices.end());
    uv_coords.erase(uv_coords.begin() + static_cast<std::ptrdiff_t>(original_vertex_count), uv_coords.end());
    faces = original_faces;
}

/*!
 * Packs the charts (faces groups) onto a texture image by using as much space as possible without having them overlap
 * @param vertices The list of vertices position
 * @param faces The list of vertices position
 * @param charts The list of grouped faces indices
 * @param uv_coords The original UV coordinates, which may be overlapping and not fitting on an image. As an output they
 *                  will be properly scaled and distributed on the image.
 * @param texture_width Output width to be used for the texture image
 * @param texture_height Output height to be used for the texture image
 * @return
 */
bool packCharts(
    const std::vector<Point3F>& vertices,
    const std::vector<Face>& faces,
    const std::vector<std::vector<size_t>>& charts,
    std::vector<Point2F>& uv_coords,
    uint32_t& texture_width,
    uint32_t& texture_height,
    const UnwrapOptions& options,
    const MeshTopology* topo = nullptr)
{
    xatlas::Atlas* atlas = xatlas::Create();
    xatlas::UvMeshDecl mesh;
    mesh.vertexUvData = uv_coords.data();
    mesh.indexData = faces.data();
    mesh.vertexCount = static_cast<uint32_t>(uv_coords.size());
    mesh.vertexStride = sizeof(Point2F);
    mesh.indexCount = faces.size() * 3;
    mesh.indexFormat = xatlas::IndexFormat::UInt32;

    if (xatlas::AddUvMesh(atlas, mesh) != xatlas::AddMeshError::Success)
    {
        xatlas::Destroy(atlas);
        spdlog::error("Error adding mesh");
        return false;
    }

    const uint32_t calculation_definition = options.pack_resolution;
    const uint32_t desired_definition = options.texture_size;

    xatlas::SetCharts(atlas, charts);

    xatlas::PackOptions pack_options{ .padding = options.padding, .resolution = calculation_definition };
    // Footprint guarantee: min_island_texels at the final texture, expressed in packing-grid texels
    pack_options.nearSearchRadius = options.near_radius;
    pack_options.nearSearchStep = options.near_step;
    pack_options.nearSearchAttempts = options.near_attempts;
    pack_options.minChartExtent = static_cast<uint32_t>(std::ceil(static_cast<double>(options.min_island_texels) * calculation_definition / std::max<uint32_t>(desired_definition, 1)));

    // Neighbour-aware placement: chart adjacency (shared 3D boundary length) and a placement order
    // that keeps neighbours consecutive (Prim-like: next = chart most connected to the placed set).
    std::vector<uint32_t> order, nb_offsets, nb_ids;
    std::vector<float> nb_weights;
    if (options.proximity_packing && topo != nullptr && ! charts.empty())
    {
        const size_t C = charts.size();
        std::vector<uint32_t> face_chart(topo->adj.size(), UINT32_MAX);
        for (size_t c = 0; c < C; ++c)
            for (const size_t f : charts[c])
                face_chart[f] = static_cast<uint32_t>(c);
        std::unordered_map<uint64_t, float> shared;
        for (size_t c = 0; c < C; ++c)
            for (const size_t f : charts[c])
                for (int s = 0; s < 3; ++s)
                {
                    const int32_t nb = s == 0 ? topo->adj[f].i1 : (s == 1 ? topo->adj[f].i2 : topo->adj[f].i3);
                    if (nb < 0)
                        continue;
                    const uint32_t cb = face_chart[nb];
                    if (cb == UINT32_MAX || cb == c)
                        continue;
                    shared[(static_cast<uint64_t>(c) << 32) | cb] += topo->edge_len[3 * f + s];
                }
        std::vector<uint32_t> degree(C, 0);
        for (const auto& [key, w] : shared)
            ++degree[key >> 32];
        nb_offsets.assign(C + 1, 0);
        for (size_t c = 0; c < C; ++c)
            nb_offsets[c + 1] = nb_offsets[c] + degree[c];
        nb_ids.assign(nb_offsets[C], 0);
        nb_weights.assign(nb_offsets[C], 0.0f);
        std::vector<uint32_t> fill(nb_offsets.begin(), nb_offsets.end() - 1);
        for (const auto& [key, w] : shared)
        {
            const uint32_t a = static_cast<uint32_t>(key >> 32);
            nb_ids[fill[a]] = static_cast<uint32_t>(key & 0xffffffffu);
            nb_weights[fill[a]] = w;
            ++fill[a];
        }
        std::vector<float> key_w(C, 0.0f);
        std::vector<uint8_t> visited(C, 0);
        std::vector<uint32_t> by_size(C);
        std::iota(by_size.begin(), by_size.end(), 0u);
        std::sort(by_size.begin(), by_size.end(), [&charts](const uint32_t a, const uint32_t b) { return charts[a].size() > charts[b].size(); });
        size_t next_seed = 0;
        std::priority_queue<std::pair<float, uint32_t>> heap;
        order.reserve(C);
        while (order.size() < C)
        {
            if (heap.empty())
            {
                while (next_seed < C && visited[by_size[next_seed]])
                    ++next_seed;
                heap.emplace(std::numeric_limits<float>::max(), by_size[next_seed]);
            }
            const auto [k, c] = heap.top();
            heap.pop();
            if (visited[c])
                continue;
            visited[c] = 1;
            order.push_back(c);
            for (uint32_t i = nb_offsets[c]; i < nb_offsets[c + 1]; ++i)
            {
                const uint32_t n = nb_ids[i];
                if (visited[n])
                    continue;
                key_w[n] += nb_weights[i];
                heap.emplace(key_w[n], n);
            }
        }
        // Cluster packing: 3D-connected groups of small charts, each packed inside one atlas block that
        // is (about) one pyramid tile of the texture fill, so gutters inside a block are filled from
        // charts of the same surface region.
        std::vector<uint32_t> group_of_chart, block_sizes;
        if (options.group_cell_texels > 0)
        {
            const uint32_t S = std::max<uint32_t>(8, static_cast<uint32_t>(std::llround(static_cast<double>(options.group_cell_texels) * calculation_definition / std::max<uint32_t>(desired_definition, 1))));
            pack_options.groupCellSize = S;
            // xatlas texel scale estimate (same formula it uses: 75% utilisation of the pack resolution)
            double mesh_area = 0.0;
            for (const Face& f : faces)
            {
                const Point2F &a = uv_coords[f.i1], &b = uv_coords[f.i2], &d = uv_coords[f.i3];
                mesh_area += 0.5 * std::abs((b.x - a.x) * (d.y - a.y) - (d.x - a.x) * (b.y - a.y));
            }
            const double tpu = std::sqrt(static_cast<double>(calculation_definition) * calculation_definition / std::max(mesh_area / 0.75, 1e-12));
            std::vector<double> chart_area(C, 0.0);
            std::vector<uint8_t> big(C, 0);
            for (size_t c = 0; c < C; ++c)
            {
                float umin = std::numeric_limits<float>::max(), umax = std::numeric_limits<float>::lowest(), vmin = umin, vmax = umax;
                for (const size_t fi : charts[c])
                    for (const uint32_t vi : { faces[fi].i1, faces[fi].i2, faces[fi].i3 })
                    {
                        umin = std::min(umin, uv_coords[vi].x); umax = std::max(umax, uv_coords[vi].x);
                        vmin = std::min(vmin, uv_coords[vi].y); vmax = std::max(vmax, uv_coords[vi].y);
                    }
                const double w = (umax - umin) * tpu + 2.0, h = (vmax - vmin) * tpu + 2.0;
                chart_area[c] = w * h;
                big[c] = (w > 0.9 * S || h > 0.9 * S) ? 1 : 0;
            }
            // Kruskal on the chart adjacency (strongest boundaries first) with an area cap per group
            std::vector<uint32_t> parent(C);
            std::iota(parent.begin(), parent.end(), 0u);
            const auto find = [&parent](uint32_t v) { while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; } return v; };
            struct Edge { float w; uint32_t a, b; };
            std::vector<Edge> edges;
            edges.reserve(nb_ids.size() / 2);
            for (uint32_t a = 0; a < C; ++a)
                for (uint32_t i = nb_offsets[a]; i < nb_offsets[a + 1]; ++i)
                    if (nb_ids[i] > a)
                        edges.push_back(Edge{ nb_weights[i], a, nb_ids[i] });
            std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) { return x.w > y.w; });
            const double cap = 0.7 * static_cast<double>(S) * S;
            for (const Edge& e : edges)
            {
                if (big[e.a] || big[e.b])
                    continue;
                const uint32_t ra = find(e.a), rb = find(e.b);
                if (ra == rb || chart_area[ra] + chart_area[rb] > cap)
                    continue;
                parent[rb] = ra;
                chart_area[ra] += chart_area[rb];
            }
            group_of_chart.assign(C, UINT32_MAX);
            std::vector<uint32_t> root_id(C, UINT32_MAX);
            uint32_t groups = 0;
            for (uint32_t c = 0; c < C; ++c)
            {
                if (big[c])
                    continue;
                const uint32_t r = find(c);
                if (root_id[r] == UINT32_MAX)
                    root_id[r] = groups++;
                group_of_chart[c] = root_id[r];
            }
            // Block size per group: the smallest power of two whose 70%-filled area holds the group
            std::vector<double> group_area(groups, 0.0);
            for (uint32_t c = 0; c < C; ++c)
                if (group_of_chart[c] != UINT32_MAX)
                    group_area[group_of_chart[c]] += chart_area[c];
            std::vector<uint32_t> block_of_chart(C, 0);
            for (uint32_t c = 0; c < C; ++c)
            {
                if (group_of_chart[c] == UINT32_MAX)
                    continue;
                const double need = std::sqrt(group_area[group_of_chart[c]] / 0.8);
                uint32_t side = static_cast<uint32_t>(std::ceil(need / 8.0)) * 8;
                block_of_chart[c] = std::clamp<uint32_t>(side, 8, S);
            }
            // Keep every group's charts consecutive in the placement order (groups appear in Prim order)
            std::vector<std::vector<uint32_t>> members(groups);
            for (const uint32_t c : order)
                if (group_of_chart[c] != UINT32_MAX)
                    members[group_of_chart[c]].push_back(c);
            // Big charts first, largest first: they dominate the area and would fragment the block grid
            // if interleaved; the groups then attach around them in Prim order.
            std::vector<uint32_t> grouped_order;
            grouped_order.reserve(C);
            for (const uint32_t c : by_size)
                if (group_of_chart[c] == UINT32_MAX)
                    grouped_order.push_back(c);
            std::vector<uint8_t> emitted(groups, 0);
            for (const uint32_t c : order)
            {
                const uint32_t g = group_of_chart[c];
                if (g != UINT32_MAX && ! emitted[g])
                {
                    emitted[g] = 1;
                    grouped_order.insert(grouped_order.end(), members[g].begin(), members[g].end());
                }
            }
            order.swap(grouped_order);
            block_sizes.swap(block_of_chart);
            spdlog::debug("uvula pack: {} groups of small charts ({} big charts), blocks 8..{} pack texels", groups, std::count(big.begin(), big.end(), 1), S);
        }
        const xatlas::ChartLayoutHints hints{ order.data(), nb_offsets.data(), nb_ids.data(), nb_weights.data(), group_of_chart.empty() ? nullptr : group_of_chart.data(), block_sizes.empty() ? nullptr : block_sizes.data(), static_cast<uint32_t>(C) };
        xatlas::SetChartLayoutHints(atlas, hints);
    }
    xatlas::PackCharts(atlas, pack_options);
    if (atlas->utilization)
        spdlog::debug("uvula pack: utilization {:.3f}", atlas->utilization[0]);

    // The texture is always square downstream: normalise both axes by the same side so a
    // non-square packed bitmap is not stretched anisotropically (the unused strip stays empty).
    const uint32_t max_side = std::max(atlas->width, atlas->height);
    texture_width = desired_definition;
    texture_height = desired_definition;
    spdlog::debug("uvula pack: bitmap {}x{} at resolution {} -> texture {}", atlas->width, atlas->height, calculation_definition, desired_definition);

    const xatlas::Mesh& output_mesh = *atlas->meshes;
    const auto side = static_cast<float>(max_side);
    for (size_t i = 0; i < output_mesh.vertexCount; ++i)
    {
        const xatlas::PlacedVertex& vertex = output_mesh.vertexArray[i];
        uv_coords[vertex.xref] = Point2F{ .x = vertex.uv[0] / side, .y = vertex.uv[1] / side };
    }

    xatlas::Destroy(atlas);
    return true;
}

static bool smartUnwrapLegacy(std::vector<Point3F>& vertices, std::vector<Face>& faces, std::vector<Point2F>& uv_coords,
                 uint32_t& texture_width, uint32_t& texture_height, std::vector<uint32_t>& vertex_remap, const UnwrapOptions& options)
{
    const std::vector<Face> original_faces(faces);
    std::vector<std::vector<size_t>> charts = makeCharts(vertices, faces, uv_coords);

    const std::vector<Face> faces_with_similar_indices = groupSimilarVertices(faces, vertices);
    charts = splitNonLinkedFacesCharts(charts, faces_with_similar_indices);

    const size_t original_vertex_count = vertices.size();

    // Phase 1: Detect and split overlapping / elongated / folded charts
    duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts);
    charts = splitSelfOverlappingCharts(charts, faces, uv_coords);
    charts = splitFoldedCharts(charts, faces, uv_coords);
    charts = splitElongatedCharts(charts, faces, uv_coords);
    restorePreDuplicationState(vertices, uv_coords, faces, original_faces, original_vertex_count);

    // Iterative merge-split: each iteration merges aggressively then runs overlap
    // detection so that folded charts are split back.  Fragments from one iteration
    // become candidates for absorption into *different* neighbors in the next.
    constexpr size_t merge_iters = 3;
    const size_t merge_thresholds[merge_iters] = {
        std::max<size_t>(original_faces.size() / 500, 32),
        std::max<size_t>(original_faces.size() / 2000, 16),
        std::max<size_t>(original_faces.size() / 8000, 4),
    };

    // Connectivity stays fixed across merge passes. Dense lookup preserves the
    // face insertion order used for neighbor counts and merge tie-breaking.
    std::vector<std::vector<size_t>> vertex_faces(original_vertex_count);
    for (size_t fi = 0; fi < faces_with_similar_indices.size(); ++fi)
    {
        const Face& face = faces_with_similar_indices[fi];
        vertex_faces[face.i1].push_back(fi);
        vertex_faces[face.i2].push_back(fi);
        vertex_faces[face.i3].push_back(fi);
    }

    for (size_t iter = 0; iter < merge_iters; ++iter)
    {
        charts = mergeTinyCharts(charts, faces_with_similar_indices, vertex_faces, original_faces, vertices, merge_thresholds[iter]);

        duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts);
        charts = splitSelfOverlappingCharts(charts, faces, uv_coords);
        charts = splitFoldedCharts(charts, faces, uv_coords);
        restorePreDuplicationState(vertices, uv_coords, faces, original_faces, original_vertex_count);
    }

    // Phase 4: Final vertex duplication and re-projection
    duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts, &vertex_remap);

    if (! packCharts(vertices, faces, charts, uv_coords, texture_width, texture_height, options))
        return false;

    // Strip degenerate faces (all 3 vertices at UV origin) and compact the vertex array.
    {
        const auto at_origin = [](const Point2F& p) { return p.x == 0.0f && p.y == 0.0f; };

        std::vector<Face> clean_faces;
        clean_faces.reserve(faces.size());
        for (const Face& f : faces)
        {
            if (at_origin(uv_coords[f.i1]) && at_origin(uv_coords[f.i2]) && at_origin(uv_coords[f.i3]))
                continue;
            clean_faces.push_back(f);
        }

        std::vector<uint32_t> idx_remap(vertices.size(), UINT32_MAX);
        std::vector<Point3F> compact_verts;
        std::vector<Point2F> compact_uvs;
        std::vector<uint32_t> compact_vremap;
        compact_verts.reserve(clean_faces.size());
        compact_uvs.reserve(clean_faces.size());
        compact_vremap.reserve(clean_faces.size());

        for (Face& f : clean_faces)
        {
            for (uint32_t* vi : { &f.i1, &f.i2, &f.i3 })
            {
                if (idx_remap[*vi] == UINT32_MAX)
                {
                    idx_remap[*vi] = static_cast<uint32_t>(compact_verts.size());
                    compact_verts.push_back(vertices[*vi]);
                    compact_uvs.push_back(uv_coords[*vi]);
                    compact_vremap.push_back(vertex_remap[*vi]);
                }
                *vi = idx_remap[*vi];
            }
        }

        vertices = std::move(compact_verts);
        uv_coords = std::move(compact_uvs);
        vertex_remap = std::move(compact_vremap);
        faces = std::move(clean_faces);
    }

    return true;
}

namespace
{

struct StageTimer
{
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    double lap()
    {
        const auto now = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(now - t).count();
        t = now;
        return s;
    }
};

std::vector<Vector3F> chartNormalsFromFaces(const std::vector<std::vector<size_t>>& charts, const std::vector<Vector3F>& face_proj)
{
    std::vector<Vector3F> normals(charts.size());
    for (size_t c = 0; c < charts.size(); ++c)
        normals[c] = face_proj[charts[c].front()];
    return normals;
}

} // namespace

bool smartUnwrap(std::vector<Point3F>& vertices, std::vector<Face>& faces, std::vector<Point2F>& uv_coords, uint32_t& texture_width, uint32_t& texture_height, std::vector<uint32_t>& vertex_remap, const UnwrapOptions& options)
{
    if (options.legacy_charting)
        return smartUnwrapLegacy(vertices, faces, uv_coords, texture_width, texture_height, vertex_remap, options);

    StageTimer timer;
    const size_t face_count = faces.size();
    const MeshTopology topo = buildTopology(vertices, faces);
    const double t_topology = timer.lap();

    Charting charting = growCharts(topo, options);
    std::vector<std::vector<size_t>> charts = chartLists(charting);
    std::vector<Vector3F> chart_normals = charting.chart_normal;
    reconnectCharts(charts, chart_normals, topo);
    const double t_charting = timer.lap();

    // Every face keeps the projection normal of the chart it was grown in, so pieces produced
    // by later splits are still projected along a direction that bounds their stretch.
    std::vector<Vector3F> face_proj(face_count);
    for (size_t c = 0; c < charts.size(); ++c)
        for (const size_t f : charts[c])
            face_proj[f] = chart_normals[c];

    const std::vector<Face> original_faces = faces;
    const size_t original_vertex_count = vertices.size();

    // Fold-free self-overlaps (helices, S-curves seen along the normal) are still possible: split them
    duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts, nullptr, &chart_normals);
    const size_t charts_before_split = charts.size();
    charts = splitSelfOverlappingCharts(charts, faces, uv_coords);
    restorePreDuplicationState(vertices, uv_coords, faces, original_faces, original_vertex_count);
    chart_normals = chartNormalsFromFaces(charts, face_proj);
    reconnectCharts(charts, chart_normals, topo);
    const double t_split = timer.lap();

    duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts, &vertex_remap, &chart_normals);
    size_t charted = 0;
    for (const auto& chart : charts)
        charted += chart.size();
    if (charted != face_count)
        spdlog::error("uvula: {} of {} faces charted", charted, face_count);

    if (! packCharts(vertices, faces, charts, uv_coords, texture_width, texture_height, options, &topo))
        return false;
    const double t_pack = timer.lap();

    size_t tiny = 0;
    for (const auto& chart : charts)
        if (chart.size() <= 2)
            ++tiny;
    spdlog::info(
        "uvula: {} charts ({} with <=2 faces, {} before overlap split) for {} faces | topology {:.2f}s charting {:.2f}s split {:.2f}s pack {:.2f}s",
        charts.size(), tiny, charts_before_split, face_count, t_topology, t_charting, t_split, t_pack);
    return true;
}
