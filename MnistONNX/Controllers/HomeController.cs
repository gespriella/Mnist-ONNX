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
        public JsonResult PredictMNIST(string alg,[FromBody] List<byte> imageBytes) {
            float[] floatArray = imageBytes.ConvertAll(x => Convert.ToSingle(x)).ToArray();
            InferenceSession inferenceSession = _inferenceSessions[alg];
            var tensor = new DenseTensor<float>(floatArray, inferenceSession.InputMetadata["input"].Dimensions);
            var results = inferenceSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", tensor) }).ToArray();

            var pred = results[0].AsTensor<long>().ToArray()[0];
            var probs = results[1].AsEnumerable<NamedOnnxValue>()
                .First().AsDictionary<long, float>().Values.ToArray();
            var WrappedReturn = new { prediction = pred, probabilities = probs };

            /*** For debugging the array ***/
            //string strArrayOfFloats="";
            //foreach (var item in floatArray){strArrayOfFloats += $"{item},";}
            //Console.Write("[" + strArrayOfFloats + "]");

            return Json(WrappedReturn);
        }

        public IActionResult Index() => View();
    }
}
