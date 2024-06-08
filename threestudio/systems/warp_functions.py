import warp as wp

@wp.func
def compute_tet_volume(x: wp.vec3, y: wp.vec3, z: wp.vec3, w: wp.vec3):
  a = y - x
  b = z - x
  c = w - x
  return wp.dot(wp.cross(a,b),c)/6.0

@wp.func
def compute_tet_com(x: wp.vec3, y: wp.vec3, z: wp.vec3):
  # assume the other point is zero
  return (x+y+z)/4.0

@wp.func
def quat2rpy(q: wp.types.quatf):
  r = wp.quat_to_matrix(q)
  pi = 3.14159265359
  r22 = r[2][2]
  r21 = r[2][1]
  r10 = r[1][0]
  r00 = r[0][0]
  rsum = wp.sqrt((r22 * r22 + r21 * r21 + r10 * r10 + r00 * r00) / 2.0)
  r20 = r[2][0]
  q2 = wp.atan2(-r20, rsum)
  e0 = q[0]
  e1 = q[1]
  e2 = q[2]
  e3 = q[3]
  yA = e1 + e3
  xA = e0 - e2
  yB = e3 - e1
  xB = e0 + e2
  epsilon = 1e-12
  isSingularA = (abs(yA) <= epsilon) and (abs(xA) <= epsilon)
  isSingularB = (abs(yB) <= epsilon) and (abs(xB) <= epsilon)
  zA = 0.0
  if not isSingularA:
    zA = wp.atan2(yA, xA)
  zB = 0.0
  if not isSingularB:
    zB = wp.atan2(yB, xB)
  q1 = zA - zB
  q3 = zA + zB
  q3 = pi - q3

  if q1 > pi:
    q1 = q1 - 2.0 * pi
  elif q1 < -pi:
    q1 = q1 + 2.0 * pi

  if q3 > pi:
    q3 = q3 - 2.0 * pi
  elif q3 < -pi:
    q3 = q3 + 2.0 * pi

  new_q = wp.vec3(q3, q2, q1)
  return new_q