
using Microsoft.ML.Data;

namespace rca.predictor.Common
{
    internal class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;

        public float[] Score;
    }
}