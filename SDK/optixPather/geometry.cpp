#include "geometry.h"

#include "GeometryData.h"

#include <array>
#include <map>

namespace geo
{
	const Vertices& Geometry::GetVertices() const { return vertices_indices.first; }
	const Normals& Geometry::GetNormals() const { return normals; }
	const VertexIndices& Geometry::GetIndices() const { return vertices_indices.second; }
	HitGroupData Geometry::GetData() const { return data; }

	Geometry::Geometry(VerticesIndices vertices_indices, HitGroupData data) :
		vertices_indices(vertices_indices),
		normals(CalculateNormals(vertices_indices)),
		data(data)
	{ }

	Normals Geometry::CalculateNormals(const VerticesIndices& vertices_indices)
	{
		Normals normals(vertices_indices.first.size(), { 0.0f, 0.0f, 0.0f });

		for (const IndexedTriangle& triangle : vertices_indices.second)
		{
			const float3 v21 = vertices_indices.first[triangle.v2] - vertices_indices.first[triangle.v1];
			const float3 v31 = vertices_indices.first[triangle.v3] - vertices_indices.first[triangle.v1];

			const float3 normal = normalize(cross(v21, v31));

			normals[triangle.v1] = normals[triangle.v1] + normal;
			normals[triangle.v2] = normals[triangle.v2] + normal;
			normals[triangle.v3] = normals[triangle.v3] + normal;
		}

		for (auto& normal : normals)
		{
			normal = normalize(normal);
		}

		return normals;
	}

	Box::Box(float3 size, sutil::Matrix4x4 transform, HitGroupData data) :
		Box(size, All, transform, data)
	{ }

	Box::Box(float3 size, unsigned char faces, sutil::Matrix4x4 transform, HitGroupData data) :
		Geometry(Make(size, faces, transform), data)
	{ }

	VerticesIndices Box::Make(float3 size, unsigned char faces, sutil::Matrix4x4 transform)
	{
		VerticesIndices v;

		v.first.emplace_back(make_float3( size.x / 2,  size.y / 2,  size.z / 2)); // 0
		v.first.emplace_back(make_float3(-size.x / 2,  size.y / 2,  size.z / 2)); // 1
		v.first.emplace_back(make_float3( size.x / 2, -size.y / 2,  size.z / 2)); // 2
		v.first.emplace_back(make_float3(-size.x / 2, -size.y / 2,  size.z / 2)); // 3
		v.first.emplace_back(make_float3( size.x / 2,  size.y / 2, -size.z / 2)); // 4
		v.first.emplace_back(make_float3(-size.x / 2,  size.y / 2, -size.z / 2)); // 5
		v.first.emplace_back(make_float3( size.x / 2, -size.y / 2, -size.z / 2)); // 6
		v.first.emplace_back(make_float3(-size.x / 2, -size.y / 2, -size.z / 2)); // 7

		for (Vertex& vertex : v.first)
		{
			vertex = transform * static_cast<float4>(vertex);
		}

		if (faces & Bottom)
		{
			v.second.push_back({ 7, 6, 2 });
			v.second.push_back({ 7, 2, 3 });
		}

		if (faces & Top)
		{
			v.second.push_back({ 5, 0, 4 });
			v.second.push_back({ 5, 1, 0 });
		}

		if (faces & Left)
		{
			v.second.push_back({ 7, 3, 1 });
			v.second.push_back({ 7, 1, 5 });
		}

		if (faces & Right)
		{
			v.second.push_back({ 6, 0, 2 });
			v.second.push_back({ 6, 4, 0 });
		}

		if (faces & Back)
		{
			v.second.push_back({ 7, 4, 6 });
			v.second.push_back({ 7, 5, 4 });
		}

		if (faces & Front)
		{
			v.second.push_back({ 3, 2, 0 });
			v.second.push_back({ 3, 0, 1 });
		}

		return v;
	}

	Icosphere::Icosphere(float radius, unsigned int subdivisions, sutil::Matrix4x4 transform, HitGroupData data) :
		Icosphere(radius, 0.0f, subdivisions, transform, data)
	{ }

	Icosphere::Icosphere(float radius, float randomness, unsigned int subdivisions, sutil::Matrix4x4 transform, HitGroupData data) :
		Geometry(Make(radius, subdivisions, randomness, transform), data)
	{ }

	VerticesIndices Icosphere::Make(float radius, unsigned int subdivisions, float randomness, sutil::Matrix4x4 transform)
	{
		constexpr float X = 0.5f;
		constexpr float Z = 0.951f;
		constexpr float N = 0.0f;

		std::vector<float3> vertices =
		{
			{ -X,  N,  Z }, // 0
			{  X,  N,  Z }, // 1
			{ -X,  N, -Z }, // 2
			{  X,  N, -Z }, // 3
			{  N,  Z,  X }, // 4
			{  N,  Z, -X }, // 5
			{  N, -Z,  X }, // 6
			{  N, -Z, -X }, // 7
			{  Z,  X,  N }, // 8
			{ -Z,  X,  N }, // 9
			{  Z, -X,  N }, // 10
			{ -Z, -X,  N }, // 11
		};

		const auto& getrad = [&radius](float var) -> float
		{
			const float r = rand() / static_cast<float>(RAND_MAX);
			return radius + (r - 0.5f) * var;
		};

		for (float3& vertex : vertices)
		{
			vertex = getrad(randomness) * normalize(vertex);
		}

		std::vector<uint3> indices =
		{
			{ 0, 4,  1 },
			{ 0, 9,  4 },
			{ 9, 5,  4 },
			{ 4, 5,  8 },
			{ 4, 8,  1 },
			{ 8, 10, 1 },
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

				vertices.push_back(getrad(randomness) * normalize(edge0 + edge1));
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

		VerticesIndices v;

		for (const float3& vertex : vertices)
		{
			v.first.emplace_back(transform * make_float4(vertex, 1.0f));
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

			for (const Vertex& n : g.GetNormals())
			{
				data.normals.push_back(n);
			}

			data.materials.push_back(g.GetData());
		}

		return data;
	}
}