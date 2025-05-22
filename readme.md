# **.NET ONNX Image Classifier**

A .NET console application that uses an ONNX (Open Neural Network Exchange) model to classify images from a specified folder and organizes them into subfolders based on their predicted categories.

## **Features**

* **Image Classification**: Classifies images using a provided ONNX model.  
* **Automatic Organization**: Moves classified images into category-named subfolders.  
* **Flexible Model Support**: Designed to work with various image classification ONNX models (e.g., SqueezeNet, ResNet, MobileNet).  
* **Customizable Labels**: Uses a plain text file for category labels.  
* **Image Format Support**: Processes common image formats like JPG, PNG, BMP.  
* **Dynamic Input/Output Detection**: Attempts to automatically determine model input/output names and dimensions.

## **Requirements**

* **.NET SDK**: 8.0
* **ONNX Model**: An .onnx image classification model file.  
* **Labels File**: A .txt file containing the class labels, one per line, corresponding to the model's output.

## **Getting Started**

### **1\. Prerequisites**

* Install the [.NET SDK](https://dotnet.microsoft.com/download) if you haven't already.

### **2\. Obtain an ONNX Model and Labels File**

You'll need an ONNX image classification model and its corresponding labels file.

* **ONNX Model**: A great place to find pre-trained models is the [ONNX Model Zoo](https://github.com/onnx/models#image-classification). Download a model like SqueezeNet, ResNet, MobileNet, etc.  
* **Labels File**: The labels file should be a plain text file where each line corresponds to a class name that the model can predict. For example, if you download an ImageNet-trained model, you'll need an ImageNet labels file (often 1000 lines).  
  Example labels.txt for a simple 3-class model:  
  cat  
  dog  
  bird

### **3\. Project Setup**

1. Clone the Repository (if applicable) or Create Project Files:  
   If you have this project in a Git repository:  
   ```
   git clone https://github.com/dimitriuz/ImageClassifierConsole 
   cd ImageClassifierConsole
   ```

   Otherwise, ensure you have the ImageClassifier.csproj and Program.cs files in a project directory.  
2. Restore Dependencies:  
   Open a terminal in the project directory and run: 
   ``` 
   dotnet restore
   ```
   This will download the necessary NuGet packages (Microsoft.ML.OnnxRuntime and SixLabors.ImageSharp).

### **4\. Running the Application**

Execute the application from the terminal, providing the paths to your image folder, ONNX model, and labels file.

dotnet run \-- "\<path\_to\_your\_image\_folder\>" "\<path\_to\_your\_model.onnx\>" "\<path\_to\_your\_labels.txt\>"

**Example:**

* On Windows:  
  dotnet run \-- "C:\\Users\\YourName\\Pictures\\UnsortedAnimals" "C:\\Models\\squeezenet1.1-7.onnx" "C:\\Models\\imagenet\_classes.txt"

* On macOS/Linux:  
  dotnet run \-- "/home/user/pictures/unsorted\_animals" "/home/user/models/squeezenet1.1-7.onnx" "/home/user/models/imagenet\_classes.txt"

The application will:

1. Scan the \<path\_to\_your\_image\_folder\> for images.  
2. For each image, predict its category using the \<path\_to\_your\_model.onnx\> and \<path\_to\_your\_labels.txt\>.  
3. Create a subfolder named after the predicted category within \<path\_to\_your\_image\_folder\> (if it doesn't exist).  
4. Move the image into the respective category subfolder.

## **How It Works**

1. **Initialization**:  
   * Loads the ONNX model using Microsoft.ML.OnnxRuntime.InferenceSession.  
   * Reads the category labels from the provided text file.  
   * Attempts to automatically determine the model's input tensor name, output tensor name, and expected input dimensions (height and width). Defaults to 224x224 if detection fails.  
2. **Image Preprocessing (ImageToTensor method)**:  
   * Loads an image using SixLabors.ImageSharp.  
   * Resizes the image to the model's expected input dimensions (e.g., 224x224). The current implementation uses ResizeMode.Stretch. For better accuracy with some models, ResizeMode.Crop or ResizeMode.Pad might be preferable (this can be adjusted in the code).  
   * Normalizes the pixel values. The current implementation uses standard ImageNet normalization values:  
     * Mean: \[0.485f, 0.456f, 0.406f\]  
     * Standard Deviation: \[0.229f, 0.224f, 0.225f\]  
   * Converts the image data into a DenseTensor\<float\> with the shape \[1, 3, height, width\] (NCHW format).  
3. **Inference (Classify method)**:  
   * Runs the model with the preprocessed image tensor.  
   * Retrieves the output tensor, which contains scores for each category.  
   * Identifies the category with the highest score.  
   * Returns the corresponding label name.  
4. **File Organization (ProcessImageFolder method)**:  
   * Iterates through image files in the input folder.  
   * Calls Classify for each image.  
   * Sanitizes the predicted category name to be a valid folder name.  
   * Creates the category subfolder if it doesn't exist.  
   * Moves the image file to the category subfolder, handling potential filename conflicts by appending a number.

## **Customization and Important Notes**

* **Model Input/Output Names**: The application tries to infer input/output tensor names. If this fails for your specific model, you may need to manually set \_inputName and \_outputName in the OnnxImageClassifier constructor. Use a tool like [Netron](https://netron.app/) to inspect your ONNX model and find these names.  
* **Preprocessing**:  
  * **Dimensions**: If the automatic dimension detection is incorrect for your model, you might need to hardcode \_targetWidth and \_targetHeight.  
  * **Normalization**: The Mean and StdDev arrays are set for models pre-trained on ImageNet. If your model was trained with different normalization parameters, update these values in OnnxImageClassifier.cs.  
  * **Tensor Format**: The code assumes an NCHW (Batch, Channels, Height, Width) input tensor format. If your model expects NHWC (Batch, Height, Width, Channels), you will need to modify the ImageToTensor method to arrange the pixel data accordingly. The constructor includes a basic check and warning for this.  
  * **Resize Mode**: The Resize operation in ImageToTensor uses ResizeMode.Stretch. Depending on how your model was trained, ResizeMode.Crop or ResizeMode.Pad might yield better results.  
* **Labels File**: Ensure the order of labels in your .txt file exactly matches the order of the output classes of your ONNX model.

## **Troubleshooting**

* **"Error: ONNX model file not found" / "Error: Labels file not found"**: Double-check the paths provided as command-line arguments. Ensure they are correct and the files exist.  
* **Incorrect Classifications**:  
  * Verify that your labels file matches the model's output.  
  * Ensure preprocessing (image size, normalization values, NCHW/NHWC format) matches what the model expects.  
  * The ONNX model itself might not be accurate enough for your specific images.  
* **Dimension Mismatches / Tensor Shape Errors**: This usually indicates that the \_targetWidth, \_targetHeight, or the tensor creation logic in ImageToTensor does not match the model's expected input shape. Use Netron to verify your model's input details.  
* **Microsoft.ML.OnnxRuntime.OnnxRuntimeException**: This can have various causes. The error message often provides clues. Common issues include:  
  * Mismatched ONNX Runtime version and model opset version.  
  * Corrupted model file.  
  * Incorrect input data type or shape.

## **License**

MIT License

Copyright (c) \[2025\] \[Dmitrii Leshchenko\]

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.