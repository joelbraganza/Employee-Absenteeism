# Clearing the environment
rm(list=ls(all=T))
# Setting working directory
setwd("C:/JOEL/important-PDFs/edwisor/Project1_EmployeeAbsenteism")
getwd() # check if it's right

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','xlsx')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

# loads the data
df = read.xlsx('Absenteeism_at_work_Project.xls', sheetIndex = 1)


# exploring the data.

# gives (row_count,column_count) of the data
dim(df)
# Viewing data
# View(df)
# give a organized(brief) Structure of the data
# str(df)
# all the column/Variable names of the dataset.
colnames(df)
# making two vectors of continuous and categorical variables.

categorical_vector= c('ID','Reason.for.absence','Month.of.absence','Day.of.the.week',
                      'Seasons','Disciplinary.failure', 'Education', 'Social.drinker',
                      'Social.smoker', 'Son', 'Pet')

numerical_vector = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
            'Work.load.Average.day.', 'Transportation.expense',
            'Hit.target', 'Weight', 'Height', 
            'Body.mass.index')
target_variable = 'Absenteeism.time.in.hours'


#Missing Values Analysis
#Creating dataframe with missing values present in each variable
missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "missing_percentage"
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)] # organizing the columns properly

#Calculating percentage of missing values of each variable
missing_val$missing_percentage = (missing_val$missing_percentage/nrow(df)) * 100

# Sorting the dataframe by percentage variable in Descending order
missing_val = missing_val[order(-missing_val$missing_percentage),]

# Saving output result into csv file
write.csv(missing_val, "Missing_perc_R.csv", row.names = F)


# deleting observations with NA values in 'Absenteeism.time.in.hours' variable and saving in new dataframe
df2 = na.omit(df,c=('Absenteeism.time.in.hours'))
dim(df2)


# creating temporary dataframe for imputing every variable seperately 
df3 = data.frame(knnImputation(df2,k=3)) # redoing knn for df2 after every variable imputation.

# I had created a for loop for mean or median or knn imputation for every continuous column but always got some
# indentation error for no reason, so had to work them out manually. Sorry for the manual long imputation.
randomIndex = sample(1:nrow(df2),1) #20
df2$Distance.from.Residence.to.Work[randomIndex] #36 # actual Value
df2$Distance.from.Residence.to.Work[randomIndex] =NA # have set it back to 36 after imputation
mean(df2$Distance.from.Residence.to.Work,na.rm =T) #29.6 mean 
median(df2$Distance.from.Residence.to.Work,na.rm =T) # 26 median 
df3$Distance.from.Residence.to.Work[randomIndex] # 36 knn in df3 (closest)
df2$Distance.from.Residence.to.Work=df3$Distance.from.Residence.to.Work # imputed with knn column of df3


randomIndex = sample(1:nrow(df2),1) #78
df2$Service.time[randomIndex] #14 # actual Value
df2$Service.time[randomIndex] =NA # have set it back to 18 after imputation
mean(df2$Service.time,na.rm =T) #12.73 mean 
median(df2$Service.time,na.rm =T) # 13 median
df3$Service.time[randomIndex] # 14 knn (closest to actual 14)
df2$Service.time=df3$Service.time # imputed with knn imputed column of df3


randomIndex = sample(1:nrow(df2),1) #279
df2$Age[randomIndex] #33 # actual Value
df2$Age[randomIndex] =NA # have set it back to 28 after imputation
mean(df2$Age,na.rm =T) #36.6 mean 
median(df2$Age,na.rm =T) # 37
df3$Age[randomIndex] # 33 knn in df2(closest to actual 33)
df2$Age=df3$Age # imputed with knn imputed column of df3


randomIndex = sample(1:nrow(df2),1) #359
df2$Work.load.Average.day.[randomIndex] #253957 # actual Value
df2$Work.load.Average.day.[randomIndex] =NA # have set it back to 246288 after imputation
mean(df2$Work.load.Average.day.,na.rm =T) #270820.5 mean 
median(df2$Work.load.Average.day.,na.rm =T) # 264249 median
df3$Work.load.Average.day.[randomIndex] # 247827.8 knn in df3 (closest)
df2$Work.load.Average.day.=df3$Work.load.Average.day.# imputed with knn(closest)


randomIndex = sample(1:nrow(df2),1) #406
df2$Transportation.expense[randomIndex] #246# actual Value
df2$Transportation.expense[randomIndex] =NA # have set it back to 284853 after imputation
mean(df2$Transportation.expense,na.rm =T) #221 mean 
median(df2$Transportation.expense,na.rm =T) # 225 median
df3$Transportation.expense[randomIndex] # 246 knn of df3 (closest)
df2$Transportation.expense =df3$Transportation.expense # imputed with knn the closest


randomIndex = sample(1:nrow(df2),1) #32
df2$Hit.target[randomIndex] #92 # actual Value
df2$Hit.target[randomIndex] =NA # have set it back to 284853 after imputation
mean(df2$Hit.target,na.rm =T) #94.7 mean 
median(df2$Hit.target,na.rm =T) # 95 median 
df3$Hit.target[randomIndex] # 92.0 knn of df3 (closest to actual 92)
df2$Hit.target=df3$Hit.target #imputed column with knn of df3 


randomIndex = sample(1:nrow(df2),1) #587
df2$Weight[randomIndex] #83 # actual Value
df2$Weight[randomIndex] =NA # have set it back to 89 after imputation
mean(df2$Weight,na.rm =T) #79.3 mean 
median(df2$Weight,na.rm =T) # 83 median 
df3$Weight[randomIndex] # 83 knn of df3(closest to actual 83)
df2$Weight=df3$Weight #imputed with knn column of df3 


randomIndex = sample(1:nrow(df2),1) #600
df2$Height[randomIndex] #171 # actual Value
df2$Height[randomIndex] =NA # have set it back to 172 after imputation
mean(df2$Height,na.rm =T) #172.1 mean 
median(df2$Height,na.rm =T) # 170 median 
df3$Height[randomIndex] # 171 knn of df2(closest to actual 171)
df2$Height=df3$Height #imputed with knn column of df2


randomIndex = sample(1:nrow(df2),1) #556
df2$Body.mass.index[randomIndex] #27 # actual Value
df2$Body.mass.index[randomIndex] =NA # have set it back to 27 after imputation
mean(df2$Body.mass.index,na.rm =T) #26.7 mean 
median(df2$Body.mass.index,na.rm =T) # 25 median 
df3$Body.mass.index[randomIndex] # 22 knn of df3(closest to actual 27)
df2$Body.mass.index=df3$Body.mass.index #imputed with knn column of df3
df3 = data.frame(knnImputation(df2,k=3))

#doing Outlier Analysis
# BoxPlots - Distribution and Outlier analysis

# Boxplot for continuous variables
for (i in 1:length(numerical_vector))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (numerical_vector[i]), x = "Absenteeism.time.in.hours"), data = subset(df2))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=numerical_vector[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box plot of absenteeism for",numerical_vector[i])))
}

# ## Plotting boxplots of continuous-variables together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)
gridExtra::grid.arrange(gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,ncol=2)


# #Remove outliers using boxplot method

#Replacing all outliers with NA
for(colN in numerical_vector)
{
  print(x=c('removing outliers for column: ',colN))
  outs = df2[,colN][df2[,colN] %in% boxplot.stats(df2[,colN])$out]
  print(x = c(('number of outliers in the column '), colN,' ',length(outs)))
  df2[,colN][df2[,colN] %in% outs] = NA
  #df2 = df[which(!df2[,colN] %in% outs),] # if to drop the observations with outliers
}





# Plotting the missing_val dataframe graph
ggplot(data = missing_val[1:18,], aes(x=reorder(Columns,-missing_percentage),y = missing_percentage))+
geom_bar(stat = "identity",fill = "blue")+xlab("Variables")+
ggtitle("Missing values percentage") + theme_bw()


# using knn to impute the msiing values from outlier-removal
# does the categorical-variable imputations as well.
df2 = data.frame(knnImputation(df2,k=3))

# use this after program shutdown (all preprocessed data)
write.csv(df2, "C:/JOEL/important-PDFs/edwisor/Project1_EmployeeAbsenteism/ImputedR.csv",)
df2 = read.csv("C:/JOEL/important-PDFs/edwisor/Project1_EmployeeAbsenteism/ImputedR.csv")





# Checking for missing value
sum(is.na(df2))


#Feature/Dimension reduction

## Correlation Plot (on numeric/continuous variables)
corrgram(df2[,numerical_vector], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
# Weight variable is highly correlated with Body.mass.index, so removing it from the dataframe
df2 = subset(df2, select = -c(Weight))

numerical_vector = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
                    'Work.load.Average.day.', 'Transportation.expense',
                    'Hit.target', 'Height', 
                    'Body.mass.index')

# doing ANOVA test over Categorical variables and the target variable.

# tried doing ANOVA in a 'for' loop below, but not working, showing error:variable lengths differ (found for 'b')
# but they have the same length
for (b in categorical_vector) {
  summary(aov(formula = Absenteeism.time.in.hours~b,data = df2))
}

# so doing it manually.
# also here, these p-values are same when outlier-removal is done first and then imputations and then in the reverse
# order too. 
summary(aov(formula = Absenteeism.time.in.hours~ID,data = df2)) #0.532
summary(aov(formula = Absenteeism.time.in.hours~Reason.for.absence,data = df2)) # 2.72e-07
summary(aov(formula = Absenteeism.time.in.hours~Month.of.absence,data = df2)) # 0.988
summary(aov(formula = Absenteeism.time.in.hours~Day.of.the.week,data = df2)) # 0.0067
summary(aov(formula = Absenteeism.time.in.hours~Seasons,data = df2)) #0.602
summary(aov(formula = Absenteeism.time.in.hours~Disciplinary.failure,data = df2)) # 0.194
summary(aov(formula = Absenteeism.time.in.hours~Education,data = df2)) # 0.248
summary(aov(formula = Absenteeism.time.in.hours~Social.drinker,data = df2)) # 0.103
summary(aov(formula = Absenteeism.time.in.hours~Social.smoker,data = df2)) # 0.249
summary(aov(formula = Absenteeism.time.in.hours~Son,data = df2)) # 0.000774
summary(aov(formula = Absenteeism.time.in.hours~Pet,data = df2)) # 0.503



# normalizing the data.
for(i in numerical_vector)
{
  print(i)
  df2[,i] = (df2[,i] - min(df2[,i]))/(max(df2[,i])-min(df2[,i]))
}


#executing the ML models over the processed, normalized data 

# using stratified sampling to divide the data into train and test
set.seed(123) # setting the seed for random indices selection, for reproducing those same results(random-indices)
# training=80%, testing=20%
train.index = sample(1:nrow(df2), 0.8 * nrow(df2)) 
train = df2[ train.index,]
test  = df2[-train.index,]

# Decision tree for classification tree
# Developing  Model on training data
DT_model = rpart(Absenteeism.time.in.hours ~., data = train, method = "anova")

#Summary of DT model
summary(DT_model)

#write rules into disk
write(capture.output(summary(DT_model)), "classification_DT_Rules.txt")
# contains all rules, helpful to visualize the trees.

#Lets predict for training data
pred_DT_train = predict(DT_model, train[,names(test) != "Absenteeism.time.in.hours"])

#Lets predict for training data
pred_DT_test = predict(DT_model,test[,names(test) != "Absenteeism.time.in.hours"])


# For training data 
print(postResample(pred = pred_DT_train, obs = train[,20]))
#      RMSE     Rsquared        MAE 
#  10.7227598  0.3619363  4.6924754 

# For testing data 
print(postResample(pred = pred_DT_test, obs = test[,20]))
#    RMSE    Rsquared         MAE 
# 14.90178731  0.06795472  5.85083585 

#Random Forest
set.seed(123)

#Develop Model on training data
RF_model = randomForest(Absenteeism.time.in.hours~., data = train)
write(capture.output(summary(RF_model)), "RF_Rules.txt")

#Lets predict for training data
pred_RF_train = predict(RF_model, train[,names(test) != "Absenteeism.time.in.hours"])

#Lets predict for testing data
pred_RF_test = predict(RF_model,test[,names(test) != "Absenteeism.time.in.hours"])

# For training data 
print(postResample(pred = pred_RF_train, obs = train[,20]))
#      RMSE  Rsquared       MAE 
# 7.1558775  0.8088047   3.0437295 

# For testing data 
print(postResample(pred = pred_RF_test, obs = test[,20]))
#       RMSE     Rsquared        MAE 
#  13.9264213    0.1732434  5.2705336 

#Linear Regression#
set.seed(123)

#Linear Regression Model on training data
LR_model = lm(Absenteeism.time.in.hours ~ ., data = train)

#making predictions for training data
pred_LR_train = predict(LR_model, train[,names(test) != "Absenteeism.time.in.hours"])

#making predictions for testing data
pred_LR_test = predict(LR_model,test[,names(test) != "Absenteeism.time.in.hours"])

# For training data 
print(postResample(pred = pred_LR_train, obs = train[,20]))
#      RMSE   Rsquared       MAE 
#  12.304108  0.159861  5.756412 

# For testing data 
print(postResample(pred = pred_LR_test, obs = test[,20]))
#       RMSE     Rsquared        MAE 
#   15.0929769  0.0424915  6.7874634

# post-principal-component analysis and feature reduction
# Creating dummy variables for categorical variables
library(mlr)
df4 = dummy.data.frame(df2, categorical_vector)
# 115 total variables created.
train.index = sample(1:nrow(df4), 0.8 * nrow(df4))
train = df4[ train.index,]
test  = df4[-train.index,]
#doing principal component analysis over the independent variables
prin_comp = prcomp(train)

#compute standard deviation of each principal component
std_dev = prin_comp$sdev

#compute variance, variance = sq.rt(std_dev)
pr_var = std_dev^2

#proportion of variance explained by the independent variables
variance_proportion = pr_var/sum(pr_var)

#cumulative scree plot for finding the precise point for finding the exact variable vs. variance structure.
plot(cumsum(variance_proportion), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")


#add a training set with principal components
train.data = data.frame(Absenteeism.time.in.hours = train$Absenteeism.time.in.hours, prin_comp$x)

# From the above plot selecting 45 components since it explains almost 95+ % data variance
train.data =train.data[,1:45]

#transform test into PCA
test.data = predict(prin_comp, newdata = test)
test.data = as.data.frame(test.data)

#select the first 45 components
test.data=test.data[,1:45]

#Decision tree for classification#

#Develop Model on training data
DT_model = rpart(Absenteeism.time.in.hours ~., data = train.data, method = "anova")
write(capture.output(summary(DT_model)), "post_dummy_PCA_DT_Rules.txt")


#Lets predict for training data
pred_DT_train = predict(DT_model, train.data)

#Lets predict for training data
pred_DT_test = predict(DT_model,test.data)


# calculate errors for training data 
print(postResample(pred = pred_DT_train, obs = train$Absenteeism.time.in.hours))
#    RMSE  Rsquared       MAE 
# 3.6177402 0.9401897 1.3132767 

# calculate errors for testing data 
print(postResample(pred = pred_DT_test, obs = test$Absenteeism.time.in.hours))
#      RMSE  Rsquared       MAE 
# 3.1980663 0.9416273 1.1111874

# Random Forest

# making Random Forest Model on training data
RF_model = randomForest(Absenteeism.time.in.hours~., data = train.data)
write(capture.output(summary(RF_model)), "post_dummy_PCA_RF_Rules.txt")

# making predictions for training data
pred_RF_train = predict(RF_model, train.data)

# making predictions for testing data
pred_RF_test = predict(RF_model,test.data)

# calculate errors training data 
print(postResample(pred = pred_RF_train, obs = train$Absenteeism.time.in.hours))
#       RMSE   Rsquared       MAE 
#  2.4585892  0.9871504   0.7784148 

# calculate errors testing data
print(postResample(pred = pred_RF_test, obs = test$Absenteeism.time.in.hours))
#      RMSE    Rsquared       MAE 
#  2.5011977  0.9276519   1.3428070 

# Linear Regression

# Linear Regression Model on training data
LR_model = lm(Absenteeism.time.in.hours ~ ., data = train.data)

# making predictions for training data
pred_LR_train = predict(LR_model, train.data)

# making predictions for testing data
pred_LR_test = predict(LR_model,test.data)

# calculate errors training data 
print(postResample(pred = pred_LR_train, obs = train$Absenteeism.time.in.hours))
#    RMSE         Rsquared          MAE 
#  0.0004333469 0.9999999991    0.0002793170 

# calculate errors testing data 
print(postResample(pred = pred_LR_test, obs =test$Absenteeism.time.in.hours))
#        RMSE       Rsquared          MAE 
#  0.0004302524   0.9999999977   0.0002896420

# the frequency distribution graph code for every variable in factors affecting the absence of employees
barplot(table(df2$ID),main="ID")
barplot(table(df2$Reason.for.absence),main="Reason.for.absence")
barplot(table(df2$Month.of.absence),main="Month.of.absence")
barplot(table(df2$Distance.from.Residence.to.Work),main="Distance.from.Residence.to.Work")
barplot(table(df2$Social.smoker),main="Social.smoker")
# and similarly for every variable.




