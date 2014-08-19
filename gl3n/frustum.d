/// Note: this module is not completly tested!
/// Use with special care, results might be wrong.

module gl3n.frustum;

private {
    import gl3n.linalg : vec4, mat4, dot;
    import gl3n.math : abs;
    import gl3n.aabb : AABB;
    import gl3n.plane : Plane;
}

///
struct Frustum {
    enum {
        LEFT, /// Used to access the planes array.
        RIGHT, /// ditto
        BOTTOM, /// ditto
        TOP, /// ditto
        NEAR, /// ditto
        FAR /// ditto
    }

    Plane[6] planes; /// Holds all 6 planes of the frustum.

    @safe pure nothrow:

    @property ref Plane left() { return planes[LEFT]; }
    @property ref Plane right() { return planes[RIGHT]; }
    @property ref Plane bottom() { return planes[BOTTOM]; }
    @property ref Plane top() { return planes[TOP]; }
    @property ref Plane near() { return planes[NEAR]; }
    @property ref Plane far() { return planes[FAR]; }

    /// Constructs the frustum from a model-view-projection matrix.
    /// Params:
    /// mvp = a model-view-projection matrix
    this(mat4 mvp) {
        mvp.transpose(); // we store the matrix row-major
        
        planes = [
            // left
            Plane(vec4(
				mvp[0][3] + mvp[0][0],
				mvp[1][3] + mvp[1][0],
				mvp[2][3] + mvp[2][0],
				mvp[3][3] + mvp[3][0])),

            // right
            Plane(vec4(
				mvp[0][3] - mvp[0][0],
				mvp[1][3] - mvp[1][0],
				mvp[2][3] - mvp[2][0],
				mvp[3][3] - mvp[3][0])),

            // bottom
            Plane(vec4(
				mvp[0][3] + mvp[0][1],
				mvp[1][3] + mvp[1][1],
				mvp[2][3] + mvp[2][1],
				mvp[3][3] + mvp[3][1])),
            // top
            Plane(vec4(
				mvp[0][3] - mvp[0][1],
				mvp[1][3] - mvp[1][1],
				mvp[2][3] - mvp[2][1],
				mvp[3][3] - mvp[3][1])),
            // near
            Plane(vec4(
				mvp[0][3] + mvp[0][2],
				mvp[1][3] + mvp[1][2],
				mvp[2][3] + mvp[2][2],
				mvp[3][3] + mvp[3][2])),
            // far
            Plane(vec4(
				mvp[0][3] - mvp[0][2],
				mvp[1][3] - mvp[1][2],
				mvp[2][3] - mvp[2][2],
				mvp[3][3] - mvp[3][2]))
        ];

        normalize();
    }

    /// Constructs the frustum from 6 planes.
    /// Params:
    /// planes = the 6 frustum planes in the order: left, right, bottom, top, near, far.
    this(Plane[6] planes) {
        this.planes = planes;
        normalize();
    }

    private void normalize() {
        foreach(ref e; planes) {
            e.normalize();
        }
    }

    /// Checks if the $(I aabb) intersects with the frustum.
    /// Returns OUTSIDE (= 0), INSIDE (= 1) or INTERSECT (= 2).
    bool intersects(AABB aabb) {
        auto hextent = aabb.halfExtent;
        auto center = aabb.center;

        foreach(plane; planes) {
            float d = dot(center, plane.p.xyz);
            float r = dot(hextent, abs(plane.p.xyz));

            if(d + r < -plane.p.w) {
                return false;
            }
        }

        return true;
    }

    /// Returns true if the $(I aabb) intersects with the frustum or is inside it.
    bool opBinaryRight(string s : "in")(AABB aabb) {
        return intersects(aabb) > 0;
    }
}