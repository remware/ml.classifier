using System;
using System.Collections.Generic;
using System.Text;

namespace rca.predictor.Common
{
    public class FullPrediction(string predictedLabel, float score, int originalSchemaIndex)
    {
        public string PredictedLabel = predictedLabel;
        public float Score = score;
        public int OriginalSchemaIndex = originalSchemaIndex;
    }
}