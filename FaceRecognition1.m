function outputLabel1=FaceRecognition1(trainPath, testPath)
    command = strrep(sprintf('python FaceRecognition1.py %s  %s', trainPath, testPath), '\', '/')
    system(command);
    load outputLabel1
    