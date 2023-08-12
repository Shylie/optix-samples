#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "optixMyPathTracer.h"

#include <vector>
#include <utility>

namespace geo
{
	using Vertices = std::vector<Vertex>;
	using Materials = std::vector<HitGroupData>;
	using MaterialIndices = std::vector<uint32_t>;
	using VertexIndices = std::vector<Index>;
	using VerticesIndices = std::pair<Vertices, VertexIndices>;

	class Geometry
	{
	public:
		const Vertices& GetVertices() const;
		const VertexIndices& GetIndices() const;
		HitGroupData GetData() const;

	protected:
		Geometry(VerticesIndices, HitGroupData);

	private:
		VerticesIndices vertices_indices;
		HitGroupData data;
	};

	class Box : public Geometry
	{
	public:
		Box(float3 center, float3 size, HitGroupData data);
	
	private:
		static VerticesIndices Make(float3 center, float3 size);
	};

	class Icosphere : public Geometry
	{
	public:
		Icosphere(float3 center, float radius, unsigned int subdivisions, HitGroupData data);

	private:
		static VerticesIndices Make(float3 center, float radius, unsigned int subdivisions);
	};

	struct GeometryData
	{
	public:
		static GeometryData MakeData(std::vector<Geometry> geometries);

		Vertices vertices;
		VertexIndices vertex_indices;
		Materials materials;
		MaterialIndices material_indices;
	};
}

#endif//GEOMETRY_H