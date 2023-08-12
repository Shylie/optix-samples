#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "optixPather.h"

#include <vector>
#include <utility>

#include <sutil/Matrix.h>

namespace geo
{
	using Vertices = std::vector<Vertex>;
	using Materials = std::vector<HitGroupData>;
	using MaterialIndices = std::vector<uint32_t>;
	using VertexIndices = std::vector<IndexedTriangle>;
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
		enum Faces : unsigned char
		{
			Bottom = 1 << 0,
			Top = 1 << 1,
			Left = 1 << 2,
			Right = 1 << 3,
			Back = 1 << 4,
			Front = 1 << 5,
			All = Bottom | Top | Left | Right | Back | Front
		};

		Box(float3 size, sutil::Matrix4x4 transform, HitGroupData data, unsigned char faces = All);
	
	private:
		static VerticesIndices Make(float3 size, sutil::Matrix4x4 transform, unsigned char faces);
	};

	class Icosphere : public Geometry
	{
	public:
		Icosphere(float radius, unsigned int subdivisions, sutil::Matrix4x4 transform, HitGroupData data);

	private:
		static VerticesIndices Make(float radius, unsigned int subdivisions, sutil::Matrix4x4 transform);
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