using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public class OnnxImageClassifier
{
    private readonly InferenceSession _session;
    private readonly List<string> _labels;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int _targetWidth;
    private readonly int _targetHeight;

    // Typical normalization values for ImageNet models
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] StdDev = { 0.229f, 0.224f, 0.225f };

    public OnnxImageClassifier(string modelPath, string labelsPath)
    {
        // Initialize ONNX session
        _session = new InferenceSession(modelPath);
        Console.WriteLine($"ONNX Model loaded from: {modelPath}");

        // It's crucial to know the exact input and output names of your model.
        // You can inspect your ONNX model using tools like Netron to find these names.
        // Common names are "data", "input", "input.1" for input and "output", "prob", "softmaxout_1" for output.
        // For this example, we assume the first input and output.
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
        
        var inputMeta = _session.InputMetadata[_inputName];
        // Assuming NCHW format [batch_size, channels, height, width]
        // Or NHWC [batch_size, height, width, channels]
        // We'll assume NCHW for many PyTorch-trained models.
        // Adjust indices if your model expects NHWC or has a different dimension order.
        if (inputMeta.Dimensions.Length == 4 && inputMeta.Dimensions[0] == 1 && inputMeta.Dimensions[1] == 3) // NCHW [1,3,H,W]
        {
            _targetHeight = inputMeta.Dimensions[2];
            _targetWidth = inputMeta.Dimensions[3];
        }
        else if (inputMeta.Dimensions.Length == 4 && inputMeta.Dimensions[0] == 1 && inputMeta.Dimensions[3] == 3) // NHWC [1,H,W,3]
        {
             _targetHeight = inputMeta.Dimensions[1];
             _targetWidth = inputMeta.Dimensions[2];
             Console.WriteLine("Warning: Model might expect NHWC format. Preprocessing is set for NCHW. Adjust if needed.");
             // If your model is NHWC, the ToTensor method needs to be adjusted.
        }
        else
        {
            // Fallback or throw error if dimensions are not as expected.
            // For many common image classification models (e.g., ResNet, SqueezeNet),
            // input dimensions are [1, 3, 224, 224] or [1, 3, 299, 299].
            Console.WriteLine($"Warning: Could not automatically determine target dimensions from model input '{_inputName}'. Assuming 224x224.");
            Console.WriteLine($"Model Input Dimensions: [{string.Join(",", inputMeta.Dimensions)}]");
            _targetWidth = 224;
            _targetHeight = 224;
        }

        Console.WriteLine($"Model expects input dimensions: {_targetWidth}x{_targetHeight}");

        // Load labels
        _labels = File.ReadAllLines(labelsPath).ToList();
        Console.WriteLine($"Labels loaded from: {labelsPath} ({_labels.Count} categories)");
    }

    private DenseTensor<float> ImageToTensor(string imagePath)
    {
        using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);

        // Resize the image to the target dimensions.
        // Using ResizeMode.Crop or Pad might be better depending on the model's training.
        // For simplicity, we use Resize.
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(_targetWidth, _targetHeight),
            Mode = ResizeMode.Stretch // Or Crop, Pad, Max. Stretch is simple but may distort.
                                      // Crop is often preferred: x.Resize(new ResizeOptions { Size = new Size(_targetWidth, _targetHeight), Mode = ResizeMode.Crop })
        }));

        // Create a tensor with dimensions [batch_size, channels, height, width]
        var tensor = new DenseTensor<float>(new[] { 1, 3, _targetHeight, _targetWidth });

        // Normalize and fill the tensor
        // Assumes NCHW format (Batch, Channels, Height, Width)
        for (int y = 0; y < _targetHeight; y++)
        {
            for (int x = 0; x < _targetWidth; x++)
            {
                Rgb24 pixel = image[x, y];
                // Normalize R, G, B channels
                tensor[0, 0, y, x] = (pixel.R / 255f - Mean[0]) / StdDev[0]; // R
                tensor[0, 1, y, x] = (pixel.G / 255f - Mean[1]) / StdDev[1]; // G
                tensor[0, 2, y, x] = (pixel.B / 255f - Mean[2]) / StdDev[2]; // B
            }
        }
        return tensor;
    }

    public string Classify(string imagePath)
    {
        var inputTensor = ImageToTensor(imagePath);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        
        var outputTensor = results.First(v => v.Name == _outputName).AsTensor<float>();

        // The output is typically a tensor of scores for each class.
        // Find the index of the class with the highest score.
        int maxIndex = 0;
        float maxScore = float.MinValue;
        for (int i = 0; i < outputTensor.Length; i++)
        {
            if (outputTensor.GetValue(i) > maxScore)
            {
                maxScore = outputTensor.GetValue(i);
                maxIndex = i;
            }
        }
        
        if (maxIndex >= 0 && maxIndex < _labels.Count)
        {
            return _labels[maxIndex];
        }
        return "Unknown"; // Should not happen if labels match model output
    }

    public void ProcessImageFolder(string folderPath)
    {
        var imageExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".gif" }; // Add more if needed
        var imageFiles = Directory.EnumerateFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly)
                                  .Where(f => imageExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                                  .ToList();

        if (!imageFiles.Any())
        {
            Console.WriteLine($"No image files found in {folderPath}");
            return;
        }

        Console.WriteLine($"Found {imageFiles.Count} images to process in {folderPath}.");

        foreach (var imageFile in imageFiles)
        {
            try
            {
                Console.Write($"Processing {Path.GetFileName(imageFile)}... ");
                string predictedCategory = Classify(imageFile);
                Console.WriteLine($"Predicted: {predictedCategory}");

                // Sanitize category name for folder creation
                string sanitizedCategory = string.Join("_", predictedCategory.Split(Path.GetInvalidFileNameChars()));
                if (string.IsNullOrWhiteSpace(sanitizedCategory)) {
                    sanitizedCategory = "Unknown_Category";
                }


                string categoryFolderPath = Path.Combine(folderPath, sanitizedCategory);
                Directory.CreateDirectory(categoryFolderPath); // Ensures directory exists

                string destFilePath = Path.Combine(categoryFolderPath, Path.GetFileName(imageFile));
                
                // Handle potential file name conflicts if moving multiple files that might end up with the same name
                // For this example, we'll overwrite. A more robust solution might rename.
                if (File.Exists(destFilePath)) {
                    // Basic conflict resolution: append a timestamp or a number
                    string originalName = Path.GetFileNameWithoutExtension(imageFile);
                    string extension = Path.GetExtension(imageFile);
                    int count = 1;
                    do
                    {
                        destFilePath = Path.Combine(categoryFolderPath, $"{originalName}_{count}{extension}");
                        count++;
                    } while (File.Exists(destFilePath));
                     Console.WriteLine($"Warning: File {Path.GetFileName(imageFile)} already exists in target. Renaming to {Path.GetFileName(destFilePath)}");
                }

                File.Move(imageFile, destFilePath);
                Console.WriteLine($"Moved to {destFilePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing {Path.GetFileName(imageFile)}: {ex.Message}");
                // Optionally, move problematic files to an "error" folder
            }
        }
        Console.WriteLine("Image classification and organization complete.");
    }

    public static void Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.WriteLine("Usage: ImageClassifier <image_folder_path> <onnx_model_path> <labels_file_path>");
            Console.WriteLine("\nExample:");
            Console.WriteLine(@"  dotnet run -- ""C:\Path\To\Your\Images"" ""C:\Path\To\Your\model.onnx"" ""C:\Path\To\Your\labels.txt""");
            Console.WriteLine("\nMake sure to replace placeholders with actual paths.");
            Console.WriteLine("You can find ONNX models (e.g., SqueezeNet, ResNet) in the ONNX Model Zoo.");
            Console.WriteLine("The labels file should be a text file with one category name per line, matching the model's output classes.");
            return;
        }

        string imageFolderPath = args[0];
        string modelPath = args[1];
        string labelsPath = args[2];

        if (!Directory.Exists(imageFolderPath))
        {
            Console.WriteLine($"Error: Image folder not found: {imageFolderPath}");
            return;
        }
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Error: ONNX model file not found: {modelPath}");
            return;
        }
        if (!File.Exists(labelsPath))
        {
            Console.WriteLine($"Error: Labels file not found: {labelsPath}");
            return;
        }

        try
        {
            var classifier = new OnnxImageClassifier(modelPath, labelsPath);
            classifier.ProcessImageFolder(imageFolderPath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An unexpected error occurred: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}

