// Copyright (c) 2026 UltiMaker
// libuvula is released under the terms of the LGPLv3 or higher.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Face.h"
#include "Point3F.h"
#include "Vector3F.h"

struct UnwrapOptions;

/*!
 * Per-face geometry and edge adjacency of the input mesh, computed once.
 * Adjacency is over position-welded vertices, so duplicated (unwelded) input vertices
 * do not break connectivity. Face normals are made orientation-consistent across each
 * connected component (a flipped input triangle gets its normal negated here only; the
 * output faces are never modified).
 */
struct MeshTopology
{
    std::vector<Face> welded; // faces with position-welded vertex indices (adjacency only)
    std::vector<FaceSigned> adj; // neighbour face across edge (i1,i2), (i2,i3), (i3,i1); -1 = none
    std::vector<Vector3F> normal; // unit normal (consistent orientation); (0,0,1) for degenerate faces
    std::vector<float> area;
    std::vector<float> edge_len; // 3 per face, same edge order as adj
    std::vector<uint8_t> degenerate; // 1 for zero-area faces (normal undefined)
};

MeshTopology buildTopology(const std::vector<Point3F>& vertices, const std::vector<Face>& faces);

/*!
 * Chart assignment: every face belongs to exactly one chart and chart_normal[c] is the
 * direction chart c is projected along. Every non-degenerate face satisfies
 * dot(face normal, chart normal) >= cos(cone half-angle), which bounds the projection
 * stretch by 1/cos and guarantees a positive (unfolded) UV orientation.
 */
struct Charting
{
    std::vector<uint32_t> face_chart;
    std::vector<Vector3F> chart_normal;
};

Charting growCharts(const MeshTopology& topo, const UnwrapOptions& options);

/*! Per-chart face lists from the labels; empty charts are dropped and chart_normal compacted to match. */
std::vector<std::vector<size_t>> chartLists(Charting& charting);

/*!
 * Splits every chart into edge-connected components over topo.adj. New charts inherit
 * the parent's normal; chart_normals stays parallel to charts.
 */
void reconnectCharts(std::vector<std::vector<size_t>>& charts, std::vector<Vector3F>& chart_normals, const MeshTopology& topo);
