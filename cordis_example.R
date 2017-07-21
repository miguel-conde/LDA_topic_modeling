# GET CORDIS DATA
myURL <- "http://cordis.europa.eu/data/cordis-h2020projects.csv"

library(data.table)
cordis_h2020projects <- fread(myURL)

## Health topics
library(dplyr)
health_projects <- filter(cordis_h2020projects, 
                          topics == "SMEInst-06-2016-2017")
dim(health_projects)

health_projects_SME1 <- filter(health_projects, fundingScheme == "SME-1" )
health_projects_SME2 <- filter(health_projects, fundingScheme == "SME-2" )

# BUILD CORPUS
library(tm)
vProy <- paste0(health_projects$title, ".", health_projects$objective)
targetProy <- paste0("A smart ecosystem for the comprehensive clinical process of managing diabetes",
                     ". ",
                     "DIASFERA aims to be the first smart ecosystem to empower physicians' and patients’ ability to optimise type I diabetes
                     treatments improving their efficiency and lowering the associated economic and social burden.
                     Practitioners who treat people with diabetes feel that the efficiency of their treatments would greatly increase if they could be
                     there all the time to monitor their effects and consequently adjust them, while encouraging their patients’ adherence to
                     healthy lifestyle habits and to follow all the clinical practice guidelines.
                     Meanwhile patients say they often find themselves alone and helpless in facing the day-to-day diabetes and miss the
                     continued support of professional caregivers.
                     Therefore, we need to amplify the physician's ability to attend patients continuously between follow up visits eliminating
                     subjectivity as much as possible and focusing on the comprehensive clinical process of managing diabetes.
                     Based on our own technology, díaSfera focuses on the comprehensive clinical practice process and is built upon cutting
                     edge statistical learning and clinical data processing techniques.")
vProy <- c(vProy, targetProy)
idxTgtProy <- length(vProy)

health_proy_corp <- VCorpus(VectorSource(vProy),
                            readerControl = list(language = "en"))

health_proy_corp <- tm_map(health_proy_corp, removePunctuation)
f <- content_transformer(function(x, pattern, s) gsub(pattern, s, x))
health_proy_corp <- tm_map(health_proy_corp, f, "['’’\"]", " ")
health_proy_corp <- tm_map(health_proy_corp, f, "[[:punct:]]+", " ")
health_proy_corp <- tm_map(health_proy_corp, content_transformer(tolower))
health_proy_corp <- tm_map(health_proy_corp, removeWords, stopwords("english"))
health_proy_corp <- tm_map(health_proy_corp, f, " s ", " ")
health_proy_corp <- tm_map(health_proy_corp, stripWhitespace)
health_proy_corp <- tm_map(health_proy_corp, stemDocument, language = "english") 


# 4 - Create a Term Document Matrix (TDM) using ‘tm’ Package.
tdm = DocumentTermMatrix(health_proy_corp) # Creating a Term document Matrix

# 5 - Create tf-idf matrix - Calculate TF-IDF i.e. Term Frequency Inverse 
#     Document Frequency for all the words in word matrix created in Step 4.
library("slam") # for row_sums()
term_tfidf <- tapply(tdm$v/row_sums(tdm)[tdm$i], tdm$j, mean) * log2(nDocs(tdm)/col_sums(tdm > 0))
summary(term_tfidf)

# 6 - Exclude all the words with tf-idf <= 0.01, to remove all the words which 
#     are less frequent.
tdm <- tdm[,term_tfidf >= 0.01]
tdm <- tdm[row_sums(tdm) > 0,]
summary(col_sums(tdm))

# 7 - Deciding best K value using Log-likelihood method - Calculate the optimal 
#     Number of topics (K) in the Corpus using log-likelihood method for the 
#     TDM calculated in Step6.
library("topicmodels") # for LDA()
best.model <- lapply(seq(2, 50, by = 1), function(d){LDA(tdm, d)})
best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik)))

# 8 - Calculating LDA - Apply LDA method using ‘topicmodels’ Package to discover 
#     topics.
max(as.numeric(best.model.logLik$V1))
k = 24;# number of topics
SEED = 786; # number of tweets used
CSC_TM <-list(VEM = LDA(tdm, k = k, 
                        control = list(seed = SEED)),
              VEM_fixed = LDA(tdm, 
                              k = k,
                              control = list(estimate.alpha = FALSE, 
                                             seed = SEED)),
              Gibbs = LDA(tdm, 
                          k = k, 
                          method = "Gibbs",
                          control = list(seed = SEED, 
                                         burnin = 1000,
                                         thin = 100, 
                                         iter = 1000)),
              CTM = CTM(tdm, 
                        k = k,
                        control = list(seed = SEED,
                                       var = list(tol = 10^-4), 
                                       em = list(tol = 10^-3))))

# 9 - Evaluate the model.
#     To compare the fitted models we first investigate the values of the models 
#     fitted with VEM and estimated and with VEM and fixed 
sapply(CSC_TM[1:2], slot, "alpha")
sapply(CSC_TM, function(x) mean(apply(posterior(x)$topics, 
                                      1, 
                                      function(z) - sum(z * log(z)))))
Topic <- topics(CSC_TM[["VEM"]], 1) # most likely topic for each document
Topic
Terms <- terms(CSC_TM[["VEM"]], 8) # most likely terms for each topic
Terms


Topic[1]
Terms[, Topic[1]]
vProy[[1]]

Topic[idxTgtProy]
Terms[, Topic[idxTgtProy]]
vProy[[idxTgtProy]]

