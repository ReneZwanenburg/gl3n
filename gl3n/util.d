/**
gl3n.util

Authors: David Herberth
License: MIT
*/

module gl3n.util;

import gl3n.linalg : Quaternion;
import gl3n.plane : PlaneT;

import std.typecons : TypeTuple;

enum isQuaternion(T)	= is(T == Quaternion!Args,	Args...);
enum isPlane(T)			= is(T == PlaneT!Args,		Args...);

unittest
{
	assert(isVector!vec2);
	assert(isVector!vec3);
	assert(isVector!vec3d);
	assert(isVector!vec4i);
	assert(!isVector!int);
	assert(!isVector!mat34);
	assert(!isVector!quat);
    
    assert(isMatrix!mat2);
    assert(isMatrix!mat34);
    assert(isMatrix!mat4);
    assert(!isMatrix!float);
    assert(!isMatrix!vec3);
    assert(!isMatrix!quat);
    
    assert(isQuaternion!quat);
    assert(!isQuaternion!vec2);
    assert(!isQuaternion!vec4i);
    assert(!isQuaternion!mat2);
    assert(!isQuaternion!mat34);
    assert(!isQuaternion!float);

    assert(isPlane!Plane);
    assert(!isPlane!vec2);
    assert(!isPlane!quat);
    assert(!isPlane!mat4);
    assert(!isPlane!float);
}

template TupleRange(int from, int to)
if (from <= to)
{
    static if (from >= to)
	{
        alias TupleRange = TypeTuple!();
    }
	else
	{
        alias TupleRange = TypeTuple!(from, TupleRange!(from + 1, to));
    }
}
