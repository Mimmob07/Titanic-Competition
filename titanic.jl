using DataFrames, CSV, Plots, DecisionTree, ScikitLearn.CrossValidation

trainData = CSV.read("data/train.csv", DataFrame)
testData = CSV.read("data/test.csv", DataFrame)
dropmissing!(trainData, "Embarked")
passengerId = testData[:, "PassengerId"]

trainData.Age = replace(trainData.Age, missing => 28)
select!(trainData, Not(["Cabin", "PassengerId", "Name", "Ticket"]))
testData.Age = replace(testData.Age, missing => 28)
select!(testData, Not(["Cabin", "PassengerId", "Name", "Ticket"]))

trainData.Embarked = Int64.(replace(trainData.Embarked, "S" => 1, "C" => 2, "Q" => 3))
trainData.Sex = Int64.(replace(trainData.Sex, "male" => 1, "female" => 2))
testData.Embarked = Int64.(replace(testData.Embarked, "S" => 1, "C" => 2, "Q" => 3))
testData.Sex = Int64.(replace(testData.Sex, "male" => 1, "female" => 2))

testData.Fare = replace(testData.Fare, missing => 14.4542)

survived = combine(groupby(trainData, "Survived"), nrow => "Count")
bar(survived.Survived, survived.Count, title = "Survivors", label = nothing, texts = survived.Count)
xticks!([0:1;], ["Dead", "Survived"]) # 0 => dead, 1 => alive

survivedBySex = combine(groupby(trainData[trainData.Survived .== 1, :], "Sex"), nrow => "Count")
bar(survivedBySex.Sex, survivedBySex.Count, title = "Survivors by gender", label = nothing, texts = survivedBySex.Count)
xticks!([1:2;], ["Male", "Female"]) # 1 => male, 2 => female

deadByPclass = combine(groupby(trainData[trainData.Survived .== 0, :], "Pclass"), nrow => "Count")
bar(deadByPclass.Pclass, deadByPclass.Count, title = "Deaths by class", label = nothing, texts = deadByPclass.Count)
xticks!([1:3;], ["1st class", "2nd class", "3rd class"])

trainY = trainData[:, "Survived"]
trainX = Matrix(trainData[:, Not(["Survived"])])

forestModel = RandomForestClassifier(n_trees = 100, max_depth = 4) # untested
fit!(forestModel, trainX, trainY)
accuracy = minimum(cross_val_score(forestModel, trainX, trainY))

predictedSurvived = predict(forestModel, Matrix(testData))

predictionDataFrame = DataFrame(PassengerId = passengerId, Survived = predictedSurvived)
CSV.write("data/submission.csv", predictionDataFrame)