%{
***************************************************************************
Viola Jones Algorithm Implementation
Authors: 
Robert Limas
William Lozada
Year: 2021
Universidad Pedagogica y Tecnologica de Colombia
***************************************************************************
%}
clear;

image = imread('Images/Face.pgm');
faces = ViolaJones(image);

imshow(image);
hold on;
[numberFaces, coordinates] = size(faces);

for face = 1: numberFaces
    rectangle('Position',[faces(face, 1), faces(face, 2), faces(face, 3), faces(face, 3)])
end
