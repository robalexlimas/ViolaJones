function [ faces ] = ViolaJones(image)
%{
%***************************************************************************
%Viola Jones Algorithm Implementation
%Authors: 
%Robert Limas
%William Lozada
%Year: 2021
%Universidad Pedagogica y Tecnologica de Colombia
%
%Inputs:
%    image (must be image gray scale)
%Outputs:
%    faces (array with face's position and size)
%
%Description:
%    This script correspond to implementation of the Viola Jones algorithm
%***************************************************************************
%}
    classifier = readClassifier('classifier.txt');
    
    faces = zeros(1, 3);
    
    scaledFactor = 0.8;
    numberFaces = 0;
    scaled = 1;
    while(true)
        integral = integralImage(image);
        imageSquared = (uint32(image)) .* (uint32(image));
        integralSquared = integralImage(imageSquared);
        facesDetected = facesDetection(classifier, integral, integralSquared);
        
        if size(facesDetected, 2) > 0
            for detected = 1 : size(facesDetected, 2)
                numberFaces = numberFaces + 1;
                faces(numberFaces, 1) = facesDetected(detected).Column / (scaledFactor ^ (scaled - 1));
                faces(numberFaces, 2) = facesDetected(detected).Row / (scaledFactor ^ (scaled - 1));
                faces(numberFaces, 3) = 24 / (scaledFactor ^ (scaled - 1));
            end
        end
        
        scaled = scaled + 1
        
        image = imresize(image, scaledFactor);
        [rows, columns] = size(image);
        
        if rows < 25 || columns < 25
            break;
        end
    end
    
end

function [ classifier ] = readClassifier(fileName) 
%{
Read the classifier file.
The firts line contains the total stages.
From the second to total stages lines contains the total features in each
stage.
In the next lines, the file contains the features.
Each feature is made up of 3 rectangles.
Eache rectangle has a line to coordinate X, coordinate Y, weight, height
and weight.
After of this values, there is a value of feature threshold and the values
of the sheets.
Finally, in the end of the features, there is the stage threshold.
This structure is repeated for the total of stages.
%}
    file = fopen(fileName);
    data = fscanf(file, '%d');
    fclose(file);
    
    stages = data(1);
    for stage = 2 : stages + 1
        features(stage - 1) = data(stage);
    end
    
    line = stages + 2;
    endStage = 0;
    for stage = 1 : stages
        for feature = 1 : features(stage)
            for rectangle = 0 : 2
                for coordinate = 1 : 4
                    coordinates(endStage + 3 * feature + rectangle - 2, coordinate) = data(line);
                    line = line + 1;
                end
                rectangleWeight(endStage + 3 * feature + rectangle - 2)= data(line);
                line = line + 1;
            end
            featureThreshold(stage, feature) = data(line);
            line = line + 1;
            alpha1(stage, feature) = data(line);
            line = line + 1;
            alpha2(stage, feature) = data(line);
            line = line + 1;
        end
        endStage = endStage + 3 * feature;
        stageThreshold(stage) = data(line);
        line = line + 1;
    end
    
    classifier.Stages = stages;
    classifier.Features = features;
    classifier.Coordinates = coordinates;
    classifier.RectangleWeight = rectangleWeight;
    classifier.FeatureThreshold = featureThreshold;
    classifier.Alpha1 = alpha1;
    classifier.Alpha2 = alpha2;
    classifier.StageThreshold = stageThreshold;
end

function [ integral ] = integralImage(image)
    [rows, columns] = size(image);
    integral = zeros(rows, columns);
    accumulator = zeros(1, columns);
    
    for row = 1 : rows
        for column = 1 : columns
           if column == 1
               accumulator(column) = double(image(row, column));
           else
               accumulator(column) = double(accumulator(column - 1)) + double(image(row, column));
           end
           if row == 1
               integral(row, column) = double(accumulator(column));
           else
               integral(row, column) = double(integral(row - 1, column)) + double(accumulator(column));
           end
        end
    end
end

function [ faces ] = facesDetection(classifier, integral, integralSquared)
    [rows, columns] = size(integral);
    
    detectedFaces = 0;
    faces(1).Column = 0;
    faces(1).Row = 0;
    for row = 1: rows - 24
        for column = 1 : columns - 24
            integralWindow = integral(row : row + 24, column : column + 24);
            squaredWindow = integralSquared(row : row + 24, column : column + 24);
            
            x2 = squaredWindow(24, 24) + squaredWindow(1, 1) - squaredWindow(1, 24) - squaredWindow(24, 1);
            x2 = x2 * 576;
            
            m = integralWindow(24, 24) + integralWindow(1, 1) - integralWindow(1, 24) - integralWindow(24, 1);
            
            variance = x2 - (m^2);
            if variance > 0
                variance = fix(sqrt(variance));
            else
                variance = 1;
            end
            
            face = violaJonesClassifier(integralWindow(1 : 25, 1 : 25), classifier, variance);
            
            if face
                detectedFaces = detectedFaces + 1;
                faces(detectedFaces).Column = column;
                faces(detectedFaces).Row = row;
            end
        end
    end
end

function [ face ] = violaJonesClassifier(integral, classifier, variance)
    face = false;
    
    index = 1;
    for stage = 1 : classifier.Stages
        stageValue = 0;
        for feature = 1 : classifier.Features(stage)
            featureValue = 0;
            for rectangle = 0 : 2
                coordinateX = classifier.Coordinates(index, 1) + 1;
                coordinateY = classifier.Coordinates(index, 2) + 1;
                rectangleWidth = classifier.Coordinates(index, 3);
                rectangleHeight = classifier.Coordinates(index, 4);
                
                cornerA = integral(coordinateY, coordinateX);
                cornerB = integral(coordinateY + rectangleHeight, coordinateX);
                cornerC = integral(coordinateY, coordinateX + rectangleWidth);
                cornerD = integral(coordinateY + rectangleHeight, coordinateX + rectangleWidth);
                
                featureValue = featureValue + (cornerD + cornerA - cornerB- cornerC) * classifier.RectangleWeight(index);
                index = index + 1;
            end
            if featureValue >= variance * classifier.FeatureThreshold(stage, feature)
                stageValue = stageValue + classifier.Alpha2(stage, feature);
            else
                stageValue = stageValue + classifier.Alpha1(stage, feature);
            end
        end
        if stageValue < 0.4 * classifier.StageThreshold(stage)
            break;
        end
    end
    if stage == classifier.Stages
        face = true;
    end
end