using Microsoft.ML;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.Data;
using rca.predictor.Common;

namespace rca.predictor
{
    // This "Labeler" class could be used in a different End-User application (Web app, other console app, desktop app, etc.)
    internal class Labeler
    {
        private readonly IConfiguration _settings; 
        private readonly string _modelPath;
        private readonly MLContext _mlContext;

        private readonly PredictionEngine<ReportedIssue, IssuePrediction> _predEngine;
        private readonly ITransformer _trainedModel;

        private FullPrediction[] _fullPredictions;

        public Labeler(string modelPath, IConfiguration settings)
        {
            _settings = settings;
            _modelPath = modelPath;
            _mlContext = new MLContext();

            // Load model from file.
            _trainedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model.
            _predEngine = _mlContext.Model.CreatePredictionEngine<ReportedIssue, IssuePrediction>(_trainedModel);
        }

        public void TestPredictionForSingleIssue()
        {
            var singleIssue = new ReportedIssue()
			{
                ID = "Any-ID",
                Title = "Crash in SqlConnection when using TransactionScope",
                Description = "I'm using SqlClient in netcoreapp2.0. Sqlclient.Close() crashes in Linux but works on Windows"
            };

            // Predict labels and scores for single hard-coded issue.
            var prediction = _predEngine.Predict(singleIssue);

            _fullPredictions = GetBestThreePredictions(prediction);

            Console.WriteLine($"==== Displaying prediction of Issue with Title = {singleIssue.Title} and Description = {singleIssue.Description} ====");

            Console.WriteLine("1st Label: " + _fullPredictions[0].PredictedLabel + " with score: " + _fullPredictions[0].Score);
            Console.WriteLine("2nd Label: " + _fullPredictions[1].PredictedLabel + " with score: " + _fullPredictions[1].Score);
            Console.WriteLine("3rd Label: " + _fullPredictions[2].PredictedLabel + " with score: " + _fullPredictions[2].Score);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        private FullPrediction[] GetBestThreePredictions(IssuePrediction prediction)
        {
            float[] scores = prediction.Score;
            int size = scores.Length;
            int index0, index1, index2 = 0;

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            _predEngine.OutputSchema[nameof(IssuePrediction.Score)].GetSlotNames(ref slotNames);

            GetIndexesOfTopThreeScores(scores, size, out index0, out index1, out index2);

            _fullPredictions =
            [
                new(slotNames.GetItemOrDefault(index0).ToString(),scores[index0],index0),
                new(slotNames.GetItemOrDefault(index1).ToString(),scores[index1],index1),
                new(slotNames.GetItemOrDefault(index2).ToString(),scores[index2],index2)
            ];

            return _fullPredictions;
        }

        private void GetIndexesOfTopThreeScores(float[] scores, int n, out int index0, out int index1, out int index2)
        {
            int i;
            float first, second, third;
            index0 = index1 = index2 = 0;
            if (n < 3)
            {
                Console.WriteLine("Invalid Input");
                return;
            }
            third = first = second = 000;
            for (i = 0; i < n; i++)
            {
                // If current element is  
                // smaller than first 
                if (scores[i] > first)
                {
                    third = second;
                    second = first;
                    first = scores[i];
                }
                // If arr[i] is in between first 
                // and second then update second 
                else if (scores[i] > second)
                {
                    third = second;
                    second = scores[i];
                }

                else if (scores[i] > third)
                    third = scores[i];
            }
            var scoresList = scores.ToList();
            index0 = scoresList.IndexOf(first);
            index1 = scoresList.IndexOf(second);
            index2 = scoresList.IndexOf(third);
        }

        // Label all issues that are not labeled yet
        public async Task LabelAllNewIssuesInRcaAnalysis()
        {
            var newIssues = await GetNewIssuesFromSettings();
            Console.WriteLine(".........Getting issue from settings.........");
            foreach (var issue in newIssues.Where(issue => issue.Title.Length > 0))
            {
                var labels = PredictLabels(issue);
                Console.WriteLine($"=============== Issue from Settings was: {issue.Title} ===============");
                Console.WriteLine("1st Label: " + labels[0].PredictedLabel + " with score: " + _fullPredictions[0].Score);
                Console.WriteLine("2nd Label: " + labels[1].PredictedLabel + " with score: " + _fullPredictions[1].Score);
                Console.WriteLine("3rd Label: " + labels[2].PredictedLabel + " with score: " + _fullPredictions[2].Score);
                
            }
        }

        private FullPrediction[] PredictLabels(ReportedIssue issue)
        {
            _fullPredictions = Predict(issue);

            return _fullPredictions;
        }

        public FullPrediction[] Predict(ReportedIssue issue)
        {
            var prediction = _predEngine.Predict(issue);

            var fullPredictions = GetBestThreePredictions(prediction);

            return fullPredictions;
        }
        
        private async Task<IReadOnlyList<ReportedIssue>> GetNewIssuesFromSettings()
        {
            var configIssues = new List<ReportedIssue>()
            {
                new()
                {
                    ID = _settings["ID"],
                    Title = _settings["Title"],
                    Description = _settings["Description"],
                }
            };

            return await Task.FromResult(configIssues);
        }
        
     }
}