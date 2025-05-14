using Microsoft.ML.Data;

namespace rca.predictor.Common
{
    //The only purpose of this class is for peek data after transforming it with the pipeline
    internal class ReportedIssue
    {
        [LoadColumn(0)]
        public string ID;
        
        // The issue area label
        [LoadColumn(1)]
        public string Area; 

        [LoadColumn(2)]
        public string Title;

        [LoadColumn(3)]
        public string Description;
    }
}