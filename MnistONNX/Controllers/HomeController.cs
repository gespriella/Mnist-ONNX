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
        private InferenceSession _inferenceSession;
        public HomeController(InferenceSession inferenceSession)
        {
            //Inference Session for MnistLR.ONNX created at Startup.cs through dependency injection
            _inferenceSession = inferenceSession;
        }

        [HttpPost]
        public JsonResult PredictMNIST(List<byte> imageBytes) {
            float[] floatArray = imageBytes.ConvertAll(x=> Convert.ToSingle(x)).ToArray();

            var tensor = new DenseTensor<float>(floatArray, _inferenceSession.InputMetadata["input"].Dimensions);
            var results = _inferenceSession.Run(new List<NamedOnnxValue>{NamedOnnxValue.CreateFromTensor("input", tensor)}).ToArray();

            var pred = results[0].AsTensor<string>().ToArray()[0];
            var probs = results[1].AsEnumerable<NamedOnnxValue>()
                .First().AsDictionary<string, float>().Values.ToArray();
            var WrappedReturn = new { prediction = pred, probabilities = probs};
            
            /*** For debugging the array ***/
            //string strArrayOfFloats="";
            //foreach (var item in floatArray){strArrayOfFloats += $"{item},";}
            //Console.Write("[" + strArrayOfFloats + "]");
            
            return Json(WrappedReturn);
        }

        public IActionResult Index() => View();
    }
}
