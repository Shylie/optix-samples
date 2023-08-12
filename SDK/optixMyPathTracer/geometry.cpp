#include "geometry.h"

#include "GeometryData.h"

#include <array>
#include <map>

namespace geo
{
	Geometry::Geometry(VerticesIndices vertices_indices, HitGroupData data) :
		vertices_indices(vertices_indices),
		data(data)
	{ }

	HitGroupData Geometry::GetData() const { return data; }
	const Vertices& Geometry::GetVertices() const { return vertices_indices.first; }
	const VertexIndices& Geometry::GetIndices() const { return vertices_indices.second; }

	Box::Box(float3 center, float3 size, HitGroupData data) :
		Geometry(Make(center, size), data)
	{ }

	VerticesIndices Box::Make(float3 center, float3 size)
	{
		VerticesIndices v;

		v.first.emplace_back(center + make_float3( size.x / 2,  size.y / 2,  size.z / 2)); // 0
		v.first.emplace_back(center + make_float3(-size.x / 2,  size.y / 2,  size.z / 2)); // 1
		v.first.emplace_back(center + make_float3( size.x / 2, -size.y / 2,  size.z / 2)); // 2
		v.first.emplace_back(center + make_float3(-size.x / 2, -size.y / 2,  size.z / 2)); // 3
		v.first.emplace_back(center + make_float3( size.x / 2,  size.y / 2, -size.z / 2)); // 4
		v.first.emplace_back(center + make_float3(-size.x / 2,  size.y / 2, -size.z / 2)); // 5
		v.first.emplace_back(center + make_float3( size.x / 2, -size.y / 2, -size.z / 2)); // 6
		v.first.emplace_back(center + make_float3(-size.x / 2, -size.y / 2, -size.z / 2)); // 7

		// bottom
		v.second.push_back({ 7, 6, 2 });
		v.second.push_back({ 7, 3, 2 });

		// top
		v.second.push_back({ 5, 4, 0 });
		v.second.push_back({ 5, 1, 0 });

		// left
		v.second.push_back({ 7, 3, 1 });
		v.second.push_back({ 7, 5, 2 });

		// right
		v.second.push_back({ 6, 2, 0 });
		v.second.push_back({ 6, 4, 0 });

		// back
		v.second.push_back({ 7, 6, 4 });
		v.second.push_back({ 7, 5, 4 });

		// front
		v.second.push_back({ 3, 2, 0 });
		v.second.push_back({ 3, 1, 0 });

		return v;
	}

	Icosphere::Icosphere(float3 center, float radius, unsigned int subdivisions, HitGroupData data) :
		Geometry(Make(center, radius, subdivisions), data)
	{ }

	VerticesIndices Icosphere::Make(float3 center, float radius, unsigned int subdivisions)
	{
		const float X = 0.525731112119133606f * radius;
		const float Z = 0.850650808352039932f * radius;
		const float N = 0.0f;

		std::vector<float3> vertices =
		{
			{ -X,  N,  Z },
			{  X,  N,  Z },
			{ -X,  N, -Z },
			{  X,  N, -Z },
			{  N,  Z,  X },
			{  N,  Z, -X },
			{  N, -Z,  X },
			{  N, -Z, -X },
			{  Z,  X,  N },
			{ -Z,  X,  N },
			{  Z, -X,  N },
			{ -Z, -X,  N },
		};

		std::vector<uint3> indices =
		{
			{ 0, 4,  1 },
			{ 0, 9,  4 },
			{ 9, 5,  4 },
			{ 4, 5,  8 },
			{ 4, 8,  1 },
			{ 8, 10, 0 },
			{ 8, 3, 10 },
			{ 5, 3,  8 },
			{ 5, 2,  3 },
			{ 2, 7,  3 },
			{ 7, 10, 3 },
			{ 7, 6, 10 },
			{ 7, 11, 6 },
			{ 11, 0, 6 },
			{ 0, 1,  6 },
			{ 6, 1, 10 },
			{ 9, 0, 11 },
			{ 9, 11, 2 },
			{ 9, 2,  5 },
			{ 7, 2, 11 }
		};

		using Lookup = std::map<std::pair<unsigned int, unsigned int>, unsigned int>;

		const auto& vertex_for_edge = [&](Lookup& lookup, unsigned int first, unsigned int second) -> unsigned int
		{
			Lookup::key_type key(first, second);
			if (key.first > key.second)
			{
				std::swap(key.first, key.second);
			}

			auto inserted = lookup.insert({ key, vertices.size() });
			if (inserted.second)
			{
				const float3& edge0 = vertices[first];
				const float3& edge1 = vertices[second];

				vertices.push_back(radius * normalize(edge0 + edge1));
			}

			return inserted.first->second;
		};

		const auto& subdivide = [&]() -> void
		{
			Lookup lookup;
			std::vector<uint3> result;

			for (auto&& tri : indices)
			{
				std::array<unsigned int, 3> mid;
				mid[0] = vertex_for_edge(lookup, tri.x, tri.y);
				mid[1] = vertex_for_edge(lookup, tri.y, tri.z);
				mid[2] = vertex_for_edge(lookup, tri.z, tri.x);

				result.push_back({ tri.x,  mid[0], mid[2] });
				result.push_back({ tri.y,  mid[1], mid[0] });
				result.push_back({ tri.z,  mid[2], mid[1] });
				result.push_back({ mid[0], mid[1], mid[2] });
			}

			indices = result;
		};

		for (int i = 0; i < subdivisions; i++)
		{
			subdivide();
		}

		for (int i = 0; i < vertices.size(); i++)
		{
			vertices[i] += center;
		}

		VerticesIndices v;

		for (const float3& vertex : vertices)
		{
			v.first.emplace_back(vertex);
		}

		for (const uint3& index : indices)
		{
			v.second.push_back(index);
		}

		return v;
	}

	GeometryData GeometryData::MakeData(std::vector<Geometry> geometries)
	{
		GeometryData data;

		for (const Geometry& g : geometries)
		{
			const size_t mat_index = data.materials.size();
			const uint3 base = make_uint3(data.vertices.size());

			for (const uint3 i : g.GetIndices())
			{
				data.vertex_indices.push_back(i + base);
				data.material_indices.push_back(mat_index);
			}

			for (const Vertex& v : g.GetVertices())
			{
				data.vertices.push_back(v);
			}

			data.materials.push_back(g.GetData());
		}

		return data;
	}
}