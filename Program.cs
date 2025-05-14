using System.Reflection;
using Microsoft.ML;
using Microsoft.Extensions.Configuration;
using rca.predictor.Common;

namespace rca.predictor;

internal static class Program
{
    private static string BaseDatasetsPath = @"Data";
    private static string DataSetPath = $"{BaseDatasetsPath}/install-issues-train.tsv";
    private static string DataSetLocation =$"{Directory.GetCurrentDirectory()}/{DataSetPath}";

    private static string BaseModelsPath = @"MLModels";
    private static string ModelRelativePath = $"{BaseModelsPath}/IssuesLabelerModel.zip";
    private static string ModelPath = $"{Directory.GetCurrentDirectory()}/{ModelRelativePath}";
    
    public enum SelectedTrainerStrategy : int { SdcaMultiClassTrainer = 1, OVAAveragedPerceptronTrainer = 2 };
    public static IConfiguration Configuration { get; set; }
    
    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 0)
        {
            var versionString = Assembly.GetEntryAssembly()?
                .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?
                .InformationalVersion;

            Console.WriteLine($"rca.predictor v{versionString}");
            Console.WriteLine("-------------");
            Console.WriteLine("\nUsage:");
            Console.WriteLine("  predictor <message>");
            return await  Task.FromResult(0);
        }
        
        SetupAppConfiguration();
        GreetCow(string.Join(' ', args));
        
        //1. ChainedBuilderExtensions and Train the model
        BuildAndTrainCow(DataSetLocation, ModelPath, SelectedTrainerStrategy.OVAAveragedPerceptronTrainer);
        
        //2. Try/test to predict a label for a single hard-coded Issue
        TestSingleLabelPrediction(ModelPath);
        
        //3. Predict Issue Labels and apply into a real GitHub repo
        await AskCowOnePredictionLabel(ModelPath);

        ConsoleHelper.ConsolePressAnyKey();
        
        return await  Task.FromResult(0);
    }
    
    
    public static void BuildAndTrainCow(string DataSetLocation, string ModelPath, SelectedTrainerStrategy selectedStrategy)
    {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<ReportedIssue>(DataSetLocation, hasHeader: true, separatorChar:'\t', allowSparse: false);
             
            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label",inputColumnName:nameof(ReportedIssue.Area))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "TitleFeaturized",inputColumnName:nameof(ReportedIssue.Title)))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "DescriptionFeaturized", inputColumnName: nameof(ReportedIssue.Description)))
                            .Append(mlContext.Transforms.Concatenate(outputColumnName:"Features", "TitleFeaturized", "DescriptionFeaturized"))
                            .AppendCacheCheckpoint(mlContext);  
                            // Use in-memory cache for small/medium datasets to lower training time. 
                            // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            // (OPTIONAL) Peek data (such as 2 records) in training DataView after applying the ProcessPipeline's transformations into "Features" 
            Common.ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 2);

            // STEP 3: Create the selected training algorithm/trainer
            IEstimator<ITransformer> trainer = null; 
            switch(selectedStrategy)
            {
                case SelectedTrainerStrategy.SdcaMultiClassTrainer:                 
                     trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
                     break;
                case SelectedTrainerStrategy.OVAAveragedPerceptronTrainer:
                {
                    // Create a binary classification trainer.
                    var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features",numberOfIterations: 10);
                    // Compose an OVA (One-Versus-All) trainer with the BinaryTrainer.
                    // In this strategy, a binary classification algorithm is used to train one classifier for each class,
                    // which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers,
                    // and choosing the prediction with the highest confidence score.
                    trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);
                        
                    break;
                }
                default:
                    ConsoleHelper.ConsoleWriteHeader("Selected strategy is not available");
                    throw new ArgumentOutOfRangeException(nameof(selectedStrategy), selectedStrategy, null);
            }

            //Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics

            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults= mlContext.MulticlassClassification.CrossValidate(data:trainingDataView, estimator:trainingPipeline, numberOfFolds: 6, labelColumnName:"Label");
                    
            ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            // STEP 5: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
            var issue = new ReportedIssue() { ID = "Any-ID", Title = "WebSockets communication is slow in my machine", Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.." };
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<ReportedIssue, IssuePrediction>(trainedModel);
            //Score
            var prediction = predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            //

            // STEP 6: Save/persist the trained model to a .ZIP file
            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            ConsoleHelper.ConsoleWriteHeader("Training process finalized");
    }
    
    private static void TestSingleLabelPrediction(string modelFilePathName)
    {
        var labeler = new Labeler(modelPath: ModelPath, Configuration);
        labeler.TestPredictionForSingleIssue();
    }
    
    private static async Task AskCowOnePredictionLabel(string ModelPath) 
    {
        Console.WriteLine();
        ConsoleHelper.ConsoleWriteHeader("Evaluating model with demo data");
        Console.WriteLine(".........Retrieving Issues from RCA analysis repo, predicting label/s and assigning it to issue from settings......");
        
        var labeler = new Labeler(ModelPath, Configuration );

        await labeler.LabelAllNewIssuesInRcaAnalysis();
        
        // labeled if fullPrediction.Score >= 0.3
        Console.WriteLine("=============== Recommended only when score is 30% or better ===============");
        Console.ReadLine();
    }
    
    private static void SetupAppConfiguration()
    {
        var builder = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json");

        Configuration = builder.Build();
    }
    
    static void GreetCow(string message)
    {
        string cow = $"\n        {message}";
        cow += @"
                              @ @
                            @@@ @
                              @@ @
               Pffii           @@@@ @@
         ___________D           @@@@@@
        |________  |            _______
         (__)   | /             \     /
         ( oo /~| |      __      \   /
         /\_|// | |     /  \     |  |
       _/____/__| |    |    |    |  |
      |       ___/-----------------_| (__)
      |      /   \     \     \       `(oo)
      |     /     \     |     |       |\/
      |    |   O  |    /     / __    /
      |----|      |-----------/  \-\\\
           \     /           | o |  \\\
    ________\___/____________\__/____\\\___________

                  RCA Express
";
        Console.WriteLine(cow);
    }
}