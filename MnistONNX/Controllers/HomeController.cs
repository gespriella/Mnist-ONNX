using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors; 

namespace MnistONNX.Controllers
{
    public class HomeController : Controller
    {
        private Dictionary<string, InferenceSession> _inferenceSessions;
        public HomeController(Dictionary<string, InferenceSession> inferenceSessions) 
        {
            _inferenceSessions = inferenceSessions;
        }

        [HttpPost]
        public JsonResult predict(string alg,[FromBody] List<byte> imageBytes) {
            if (alg == "cnn") return predictCNN(imageBytes);

            // Convert bytes to floats
            float[] floatArray = imageBytes.ConvertAll(x => Convert.ToSingle(x)).ToArray();

            // Point to the correct Algorithm
            InferenceSession inferenceSession = _inferenceSessions[alg];

            // Create DenseTensor with correct dimensions
            var tensor = new DenseTensor<float>(floatArray, inferenceSession.InputMetadata["input"].Dimensions);

            // Run Prediction
            var results = inferenceSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", tensor) }).ToArray(); 

            var pred = results[0].AsTensor<long>().ToArray()[0]; // The prediction number
            var probs = results[1].AsEnumerable<NamedOnnxValue>() // The probabilities array
                .First().AsDictionary<long, float>().Values.ToArray();
            var WrappedReturn = new { prediction = pred, probabilities = probs };

            return Json(WrappedReturn);
        }

        [HttpPost]
        public JsonResult predictCNN([FromBody] List<byte> imageBytes)
        {
            float[] floatArray = imageBytes.Select(i => Convert.ToSingle(i/255.0)).ToArray();
            var matrix = floatArray.ToTensor().Reshape(new[] { 28, 28 });
            InferenceSession inferenceSession = _inferenceSessions["cnn"];
            var tensor = new DenseTensor<float>(floatArray, inferenceSession.InputMetadata["Input3"].Dimensions);
            var results = inferenceSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("Input3", tensor) }).ToArray();
            var weights = results[0].AsTensor<float>().ToList();
            var probs = weights.Select(x => x + Math.Abs(weights.Min()));
            probs = probs.Select(x => x / probs.Sum()).ToArray();
            var pred = probs.Select((n, i) => (Number: n, Index: i)).Max().Index;
            var WrappedReturn = new { prediction = pred, probabilities = probs };

            return Json(WrappedReturn);
        }

        public IActionResult Index() => View();
    }
}
