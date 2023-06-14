library(tidyverse)
exams <- read.csv("/kaggle/input/student-performance-in-mathematics/exams.csv")
head(exams, 5)
tail(exams, 5)
str(exams)
colnames(exams)
class(exams)
summary(exams)
standard_deviation_math_score <- sd(exams$math.score)
standard_deviation_writing_score <- sd(exams$writing.score)
standard_deviation_reading_score <- sd(exams$reading.score)
is.na(exams)
which(is.na(exams))
exam_scores <- na.omit(exams)
dim(exam_scores)
dim(exams)
