I = imread('circuit.tif');
offsets = [0 1; -1 1; -1 0; -1 -1];
GLCM2 = graycomatrix(I,'NumLevels',8,'Offset',offsets);
stats = GLCM_Features(GLCM2);
