import vtk

# Create a cylinder source
cylinder = vtk.vtkCylinderSource()
cylinder.SetHeight(2.0)
cylinder.SetRadius(1.0)
cylinder.SetResolution(50)

# Convert the generated mesh to the VTK Unstructured Grid format
output = vtk.vtkUnstructuredGrid()
output.ShallowCopy(cylinder.GetOutput())

# Get the point coordinates from the output and create a new vtkPoints object
points = output.GetPoints()
new_points = vtk.vtkPoints()
new_points.SetNumberOfPoints(points.GetNumberOfPoints())

# Loop over the point coordinates and convert them to double precision
for i in range(points.GetNumberOfPoints()):
    pt = points.GetPoint(i)
    new_pt = (pt[0], pt[1], pt[2])
    new_points.SetPoint(i, new_pt)

# Set the new points for the unstructured grid
output.SetPoints(new_points)

# Write the unstructured grid to a VTK file
writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName("cylinder.vtk")
writer.SetInputData(output)
writer.Write()
