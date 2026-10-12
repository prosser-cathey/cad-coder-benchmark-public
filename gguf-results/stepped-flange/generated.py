import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.75, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.7578947368421053, 0.0).circle(0.7578947368421053)
loop1=wp_sketch0.moveTo(0.7578947368421053, 0.0).circle(0.37894736842105264)
solid0=wp_sketch0.add(loop0).add(loop1).extrude(0.1484375)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.375, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop2=wp_sketch1.moveTo(0.37894736842105264, 0.0).circle(0.37894736842105264)
loop3=wp_sketch1.moveTo(0.37894736842105264, 0.0).circle(0.18947368421052632)
solid1=wp_sketch1.add(loop2).add(loop3).extrude(0.1484375)
solid=solid.union(solid1)
# Generating a workplane for sketch 2
wp_sketch2 = cq.Workplane(cq.Plane(cq.Vector(-0.375, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop4=wp_sketch2.moveTo(0.37894736842105264, 0.0).circle(0.37894736842105264)
loop5=wp_sketch2.moveTo(0.37894736842105264, 0.0).circle(0.18947368421052632)
solid2=wp_sketch2.add(loop4).add(loop5).extrude(0.375)
solid=solid.union(solid2)
# Generating a workplane for sketch 3
wp_sketch3 = cq.Workplane(cq.Plane(cq.Vector(-0.1875, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop6=wp_sketch3.moveTo(0.18947368421052632, 0.0).circle(0.18947368421052632)
solid3=wp_sketch3.add(loop6).extrude(0.375)
solid=solid.union(solid3)
# Generating a workplane for sketch 4
wp_sketch4 = cq.Workplane(cq.Plane(cq.Vector(-0.1875, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop7=wp_sketch4.moveTo(0.18947368421052632, 0.0).circle(0.18947368421052632)
solid4=wp_sketch4.add(loop7).extrude(0.75)
solid=solid.union(solid4)
