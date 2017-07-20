# http://www.bigdatanews.datasciencecentral.com/profiles/blogs/topic-modeling-in-r
# http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/

library("tm")
library("wordcloud")
library("slam")
library("topicmodels")

# Load Text
# con <- file("tweets.txt", "rt")
# tweets = readLines(con)

# 1 - Fetch tweets data using ‘twitteR’ package.
pacman::p_load(twitteR)
# ... accessing twitter, hidden
# setup_twitter_oauth()
tweets = searchTwitter("#spain", n = 100)
as.data.frame(tweets)

# 2 - Load the data into the R environment.
tweets = lapply(tweets, function(x) return(x$text))

# 3 - Clean Text - Clean the Data to remove: re-tweet information, links, 
#     special characters, emoticons, frequent words like is, as, this etc.
tweets = gsub("(RT|via)((?:\\b\\W*@\\w+)+)","",tweets)
tweets = gsub("http[^[:blank:]]+", "", tweets)
tweets = gsub("@\\w+", "", tweets)
tweets = gsub("[ \t]{2,}", "", tweets)
tweets = gsub("^\\s+|\\s+$", "", tweets)
tweets <- gsub('\\d+', '', tweets)
tweets = gsub("[[:punct:]]", " ", tweets)

# 4 - Create a Term Document Matrix (TDM) using ‘tm’ Package.
corpus = Corpus(VectorSource(tweets))
corpus = tm_map(corpus,removePunctuation)
corpus = tm_map(corpus,stripWhitespace)
corpus = tm_map(corpus,tolower)
corpus = tm_map(corpus,removeWords,stopwords("english"))
tdm = DocumentTermMatrix(corpus) # Creating a Term document Matrix

# 5 - Create tf-idf matrix - Calculate TF-IDF i.e. Term Frequency Inverse 
#     Document Frequency for all the words in word matrix created in Step 4.
term_tfidf <- tapply(tdm$v/row_sums(tdm)[tdm$i], tdm$j, mean) * log2(nDocs(tdm)/col_sums(tdm > 0))
summary(term_tfidf)

# 6 - Exclude all the words with tf-idf <= 0.1, to remove all the words which 
#     are less frequent.
tdm <- tdm[,term_tfidf >= 0.1]
tdm <- tdm[row_sums(tdm) > 0,]
summary(col_sums(tdm))

# 7 - Deciding best K value using Log-likelihood method - Calculate the optimal 
#     Number of topics (K) in the Corpus using log-likelihood method for the 
#     TDM calculated in Step6.
best.model <- lapply(seq(2, 50, by = 1), function(d){LDA(tdm, d)})
best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik)))

# 8 - Calculating LDA - Apply LDA method using ‘topicmodels’ Package to discover 
#     topics.
k = 50;# number of topics
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


Topic[31]
Terms[, Topic[31]]
inspect(corpus[[31]])
