// Copyright (c) 2025 UltiMaker
// libuvula is released under the terms of the LGPLv3 or higher.

#pragma once

#include <cstdint>
#include <vector>

#include "Face.h"

class Point3F;
struct Point2F;

/*! Tunables of smartUnwrap. Angles in degrees, sizes in texels of the given texture size. */
struct UnwrapOptions
{
    float cone_half_angle_deg{ 40.0f }; // max angle between a face normal and its chart's projection normal (size vs flatness knob)
    float absorb_half_angle_deg{ 75.0f }; // relaxed cone under which tiny charts are absorbed by a neighbour (< 90: never folds)
    int refine_iterations{ 3 }; // Lloyd-style re-flood passes of the charter
    int smoothing_passes{ 2 }; // boundary majority smoothing passes per refine iteration
    int min_island_faces{ 3 }; // islands with at most this many faces go through the small-island policy
    int min_island_texels{ 6 }; // minimum footprint (short side) of a standalone island
    uint32_t pack_resolution{ 1024 }; // xatlas packing grid; placement quantum = texture_size / pack_resolution texels
    uint32_t texture_size{ 4096 }; // texture the UVs are meant for
    uint32_t padding{ 0 }; // xatlas chart padding, in packing-grid texels
    bool proximity_packing{ true }; // place charts next to their 3D neighbours
    uint32_t near_radius{ 96 }; // ring search radius around placed neighbours, packing-grid texels
    uint32_t near_step{ 2 }; // ring stride
    uint32_t near_attempts{ 2048 }; // canBlit budget per chart before falling back to the generic search
    uint32_t group_cell_texels{ 0 }; // cluster packing: block size in texture texels (0 = off; costs ~35% density for a modest locality gain)
    bool legacy_charting{ false }; // previous normal-bucket pipeline, for A/B comparison
};

/*!
 * Groups, projects and packs the faces of the input mesh to non-overlapping, properly distributed UV patches.
 * Vertices shared by several charts are duplicated; vertex_remap maps every output vertex to its input vertex.
 * No input face is ever dropped.
 */
bool smartUnwrap(std::vector<Point3F>& vertices, std::vector<Face>& faces, std::vector<Point2F>& uv_coords,
                 uint32_t& texture_width, uint32_t& texture_height, std::vector<uint32_t>& vertex_remap,
                 const UnwrapOptions& options = {});
