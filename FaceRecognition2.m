function outputLabel2=FaceRecognition2(trainPath, testPath)
    command = strrep(sprintf('python FaceRecognition2.py %s  %s', trainPath, testPath), '\', '/')
    system(command);
    load outputLabel2
    