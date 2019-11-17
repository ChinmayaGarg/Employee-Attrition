my_data<-read.csv("C:/Users/chinm/Desktop/BIM/employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv", header = TRUE)

#cor(my_data, use = "everything")

summary(my_data)
str(my_data)

my_data1<-read.csv("C:/Users/chinm/Desktop/BIM/employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition - Copy.csv", header = TRUE)

my_data1$CAttrition[my_data1$Attrition=="Yes"]<-1
my_data1$CAttrition[my_data1$Attrition=="No"]<-0

library(GGally)

ggcorr(my_data1, 
       label = TRUE, 
       label_alpha = TRUE)

#my_data1$DAttrition[my_data1$Attrition=="Yes"]<-0
#my_data1$DAttrition[my_data1$Attrition=="No"]<-1

ggcorr(my_data1, 
       label = TRUE, 
       label_alpha = TRUE)

my_data1$CBusinessTravel[my_data1$BusinessTravel=="Non-Travel"]<-0
my_data1$CBusinessTravel[my_data1$BusinessTravel=="Travel_Frequently"]<-1
my_data1$CBusinessTravel[my_data1$BusinessTravel=="Travel_Rarely"]<-2

my_data1$CGender[my_data1$Gender=="Male"]<-1
my_data1$CGender[my_data1$Gender=="Female"]<-0

my_data1$COverTime[my_data1$OverTime=="Yes"]<-1
my_data1$COverTime[my_data1$OverTime=="No"]<-0

ggcorr(my_data1[,36:39], 
       label = TRUE, 
       label_alpha = TRUE)

#There has been no or weak correlation between employee attrition and the given parameters.
#Parameters not checked independently are Department, EducationField and JobRole.
#There  is high probability that combination of more than one parameter affects employee attrition.

