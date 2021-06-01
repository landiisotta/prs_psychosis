require(data.table)
require(caret)

set.seed(42)

# Data imputation function
impute_data <- function(train, test, c_feat, cat_feat) {
  for (k in names(train)) {
    if (k %in% c_feat) {
      # impute numeric variables with median
      med <- median(train[[k]], na.rm = T)
      set(x = train, which(is.na(train[[k]])), k, med)
      set(x = test, which(is.na(test[[k]])), k, med)

    } else if (k %in% cat_feat) {
      ## impute categorical variables with mode
      mode <- names(which.max(table(train[[k]])))
      set(x = train, which(is.na(train[[k]])), k, mode)
      set(x = test, which(is.na(test[[k]])), k, mode)
    }
  }
  return(list(train, test))
}


# Train/test split (70/30)
trts_split <- function(df, strat) {
  df.index <- createDataPartition(strat, p = .7, list = FALSE)

  train <- df[df.index]
  test <- df[-df.index]

  return(list(train, test))
}


# Create dummy variables
create_dummy <- function(df, col) {
  idx <- which(names(df) == col)
  df_first = fastDummies::dummy_cols(df[, 1:idx],
                                     select_columns = col,
                                     remove_selected_columns = TRUE,
                                     remove_first_dummy = TRUE)
  df_last = df[, (idx + 1):ncol(df)]
  return(cbind(df_first, df_last))
}


# Create and save nested datasets, i.e., w/ clinical, genetic, clinical+genetic features
create_nested <- function(train, test, name, catvar, genvar){
  fwrite(train[, ..catvar], 
         paste0("../out/biome_clinical_", sprintf("%s", name), "_train.txt", sep=""), 
         sep="\t")
  fwrite(test[, ..catvar], 
         paste0("../out/biome_clinical_", sprintf("%s", name), "_test.txt", sep=""), 
         sep="\t")
  
  fwrite(train, 
         paste0("../out/biome_all_", sprintf("%s", name), "_train.txt", sep=""), 
         sep="\t")
  fwrite(test, 
         paste0("../out/biome_all_", sprintf("%s", name), "_test.txt", sep=""), 
         sep="\t")
  
  geneticvar <- c(genvar, "gill.ContinentalGrouping", catvar[length(catvar)])
  fwrite(train[, ..geneticvar], 
         paste0("../out/biome_genetic_", sprintf("%s", name), "_train.txt", sep=""),
         sep="\t")
  fwrite(test[, ..geneticvar], 
         paste0("../out/biome_genetic_", sprintf("%s", name), "_test.txt", sep=""), 
         sep="\t")
}

##############
## Run code ##
##############

# Read Biome dataset
biome <- fread('../data/biome_dataset.txt', sep = '\t')
biome <- biome[scz.icd.binary == 1]

# Create features and response variable dataframes
dep_var <- c("NLP.agressive", "biomeOrd.PsychAdmit")
gen_var <- names(biome)[grep("PC[0-9]|PRS|gill.ContinentalGrouping", names(biome))]
indep_var <- setdiff(names(biome), c(dep_var, gen_var, "scz.icd.binary"))

cols <- indep_var
biome_df <- cbind(biome[, ..cols], biome[, ..gen_var])
for (k in names(biome_df)){
  set(biome_df, which(biome_df[,..k]==""), k, NA)
}

biome_df$biomeQ.adopt[which(biome_df$biomeQ.adopt == "No")] <- 0
biome_df$biomeQ.adopt[which(biome_df$biomeQ.adopt == "Yes")] <- 1
biome_df$biomeQ.adopt[which(biome_df$biomeQ.adopt == "Unknown")] <- NA

biome_df$biomeQ.smoke[which(biome_df$biomeQ.smoke == "No")] <- 0
biome_df$biomeQ.smoke[which(biome_df$biomeQ.smoke == "Yes")] <- 1
biome_df$biomeQ.smoke[which(biome_df$biomeQ.smoke == "Unknown")] <- NA

biome_df$biomeQ.aut.grandpar[which(biome_df$biomeQ.aut.grandpar > 1)] <- 1
biome_df$biomeQ.scz.grandpar[which(biome_df$biomeQ.scz.grandpar > 1)] <- 1
biome_df$biomeQ.dep.grandpar[which(biome_df$biomeQ.dep.grandpar > 1)] <- 1

biome_df$biomeDem.relationship[which(biome_df$biomeDem.relationship == "SINGLE")] <- 0
biome_df$biomeDem.relationship[which(biome_df$biomeDem.relationship == "WIDOWED")] <- 1
biome_df$biomeDem.relationship[which(biome_df$biomeDem.relationship == "DIVORCED/SEPARATED")] <- 2
biome_df$biomeDem.relationship[which(biome_df$biomeDem.relationship == "PARTNERED")] <- 3

biome_df <- biome_df[, (indep_var) := lapply(.SD, as.factor), .SDcols = indep_var]

biome_out <- biome[, ..dep_var]
biome_out <- biome_out[, (dep_var) := lapply(.SD, as.factor), .SDcols = dep_var]

# Create input dataframe and remove outcome NAs
biome_aggressive <- cbind(biome_df[, -"gill.ContinentalGrouping"], 
                          "gill.ContinentalGrouping" = biome_df$gill.ContinentalGrouping, 
                          "NLP.agressive" = biome_out$NLP.agressive)
biome_aggressive <- biome_aggressive[!is.na(NLP.agressive)]
biome_aggressive_na <- apply(biome_aggressive,
                             MARGIN = 2, function(x) sum(is.na(x)) / length(x))
biome_aggressive_na_dropcols <- names(biome_aggressive_na[biome_aggressive_na > 0.7])
biome_aggressive_na_dropcols
strat_aggressive <- paste(biome_aggressive$NLP.agressive,
                          biome_aggressive$gill.ContinentalGrouping,
                          sep = '-')
ag <- trts_split(biome_aggressive,
                 strat = strat_aggressive)

# Merge all patients with >=1 admissions into the same class (binarize)
biome_admit <- cbind(biome_df[, -"gill.ContinentalGrouping"], 
                     "gill.ContinentalGrouping" = biome_df$gill.ContinentalGrouping, 
                     "biomeOrd.PsychAdmit" = biome_out$biomeOrd.PsychAdmit)
biome_admit <- biome_admit[!is.na(biomeOrd.PsychAdmit)]
biome_admit_na <- apply(biome_admit,
                        MARGIN = 2, function(x) sum(is.na(x)) / length(x))
biome_admit_na_dropcols <- names(biome_admit_na[biome_admit_na > 0.7])
biome_admit_na_dropcols
biome_admit$biomeOrd.PsychAdmit <- as.numeric(as.character(biome_admit$biomeOrd.PsychAdmit))
biome_admit$biomeOrd.PsychAdmit[which(biome_admit$biomeOrd.PsychAdmit > 1)] <- 1
biome_admit$biomeOrd.PsychAdmit <- as.factor(biome_admit$biomeOrd.PsychAdmit)
strat_admit <- paste(biome_admit$biomeOrd.PsychAdmit,
                     biome_admit$gill.ContinentalGrouping,
                     sep = '-')
ad <- trts_split(biome_admit, strat = strat_admit)

# Impute datasets
c_feat <- names(ag[[1]])[grep("PC[0-9]{1,2}|PRS", names(ag[[1]]))]
cat_feat <- names(ag[[1]])[which(!names(ag[[1]]) %in% c(c_feat, 
                                                        "gill.ContinentalGrouping"))]
ag_eur <- impute_data(ag[[1]][gill.ContinentalGrouping == "EUR"],
                      ag[[2]][gill.ContinentalGrouping == "EUR"],
                      c_feat, cat_feat)
ag_amr <- impute_data(ag[[1]][gill.ContinentalGrouping == "AMR"],
                      ag[[2]][gill.ContinentalGrouping == "AMR"],
                      c_feat, cat_feat)
ag_afr <- impute_data(ag[[1]][gill.ContinentalGrouping == "AFR"],
                      ag[[2]][gill.ContinentalGrouping == "AFR"],
                      c_feat, cat_feat)
ag <- list(rbind(ag_eur[[1]], ag_afr[[1]], ag_amr[[1]]),
           rbind(ag_eur[[2]], ag_afr[[2]], ag_amr[[2]]))

c_feat <- names(ad[[1]])[grep("PC[0-9]{1,2}|PRS", names(ad[[1]]))]
cat_feat <- names(ad[[1]])[which(!names(ad[[1]]) %in% c(c_feat, 
                                                        "gill.ContinentalGrouping"))]
ad_eur <- impute_data(ad[[1]][gill.ContinentalGrouping == "EUR"],
                      ad[[2]][gill.ContinentalGrouping == "EUR"],
                      c_feat, cat_feat)
ad_afr <- impute_data(ad[[1]][gill.ContinentalGrouping == "AFR"],
                      ad[[2]][gill.ContinentalGrouping == "AFR"],
                      c_feat, cat_feat)
ad_amr <- impute_data(ad[[1]][gill.ContinentalGrouping == "AMR"],
                      ad[[2]][gill.ContinentalGrouping == "AMR"],
                      c_feat, cat_feat)
ad <- list(rbind(ad_eur[[1]], ad_afr[[1]], ad_amr[[1]]),
           rbind(ad_eur[[2]], ad_afr[[2]], ad_amr[[2]]))

ag_train <- create_dummy(ag[[1]], "biomeDem.relationship")
ag_test <- create_dummy(ag[[2]], "biomeDem.relationship")

ad_train <- create_dummy(ad[[1]], "biomeDem.relationship")
ad_test <- create_dummy(ad[[2]], "biomeDem.relationship")

# Save list of features to scale
rescalevar <- names(ag_train)[grep("PRS|PC", names(ag_train))]
fwrite(data.table("feat" = rescalevar),
       '../out/rescale_feature_biome.txt')


# Save categorical features ids
catidx <- which(names(ag_train)[1:(length(names(ag_train)) - 1)]
                %in% c(cat_feat,
                       names(ag_train)[grep("_[0-9]",
                                            names(ag_train))])) - 1
fwrite(data.table("idx" = catidx),
       '../out/categorical_feature_idx_biome.txt')

# Save outputs
create_nested(ag_train, ag_test,
              name = "agressive",
              names(ag_train)[-grep("PC|PRS", names(ag_train))],
              rescalevar)
create_nested(ad_train, ad_test,
              name = "psych_admit",
              names(ad_train)[-grep("PC|PRS", names(ad_train))],
              rescalevar)

