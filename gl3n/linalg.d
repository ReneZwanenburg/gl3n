/**
gl3n.linalg

Special thanks to:
$(UL
  $(LI Tomasz Stachowiak (h3r3tic): allowed me to use parts of $(LINK2 https://bitbucket.org/h3r3tic/boxen/src/default/src/xf/omg, omg).)
  $(LI Jakob Øvrum (jA_cOp): improved the code a lot!)
  $(LI Florian Boesch (___doc__): helps me to understand opengl/complex maths better, see: $(LINK http://codeflow.org/).)
  $(LI #D on freenode: answered general questions about D.)
)

Authors: David Herberth
License: MIT

Note: All methods marked with pure are weakly pure since, they all access an instance member.
All static methods are strongly pure.
*/


module gl3n.linalg;

import std.math : isFinite;
import std.conv : to;
import std.traits : isIntegral, isFloatingPoint, isStaticArray, isDynamicArray, isImplicitlyConvertible, isArray;
import std.string : format, rightJustify;
import std.array : join;
import std.algorithm : max, min, reduce, all, among;
import std.functional : binaryFun;
import gl3n.math : clamp, PI, sqrt, sin, cos, acos, tan, asin, atan2, almostEqual;
import gl3n.util : TupleRange;
import gl3n.vector;
import gl3n.matrix;
