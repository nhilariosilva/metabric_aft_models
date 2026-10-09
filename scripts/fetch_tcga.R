# Fetches TCGA-BRCA clinical + mutation data via TCGAbiolinks/maftools and writes
# the train/test split. Originally `KM_survival_BRCA.R`; the only change here is
# that the output directory is a CLI argument instead of a hardcoded path.
#
# Usage: Rscript scripts/fetch_tcga.R <output_dir>

# args <- commandArgs(trailingOnly = TRUE)
# if (length(args) < 1) stop("usage: Rscript fetch_tcga.R <output_dir>")
# out_dir <- args[1]
# dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

library(TCGAbiolinks)
library(tidyverse)
library(maftools)

# Pulls demographics, staging, and survival data
clinical_base <- GDCquery_clinic(project = "TCGA-BRCA", type = "clinical") %>%
  as_tibble()

# Select only a few columns from the major clinical data table:
# Response time variables
# Patient's age at index
clinical_selected <- clinical_base %>%
  select(
    submitter_id,
    days_to_death,
    days_to_last_follow_up,
    age_at_index,
    # Size and extent of primary tumor
    ajcc_pathologic_t,
    # Degree of spread to regional lymph nodes
    ajcc_pathologic_n,
    # Presence of distant metastasis
    # ajcc_pathologic_m,
    # Patient received systemic drug therapy (yes or no)
    treatments_pharmaceutical_treatment_or_therapy,
    # Patient received radiation therapy (yes or no)
    treatments_radiation_treatment_or_therapy,
    # If received radiation therapy, specifies the style
    treatments_radiation_treatment_type
  ) %>%
  mutate(
    treatments_pharmaceutical_treatment_or_therapy = replace_na(treatments_pharmaceutical_treatment_or_therapy, "unknown"),
    treatments_radiation_treatment_or_therapy = replace_na(treatments_radiation_treatment_or_therapy, "unknown"),
    treatments_radiation_treatment_type = replace_na(treatments_radiation_treatment_type, "unknown")
  ) %>%
  mutate(
    treatments_radiation_treatment_type = case_when(
      treatments_radiation_treatment_type == "Radiation Therapy, NOS" ~ "radiation_therapy_NOS",
      treatments_radiation_treatment_type == "Radiation, External Beam" ~ "radiation_external_beam",
      treatments_radiation_treatment_type == "unknown" ~ "unknown",
    ),
    ajcc_pathologic_t = case_when(
      ajcc_pathologic_t == "T1" ~ "T1",
      ajcc_pathologic_t == "T1a" ~ "T1",
      ajcc_pathologic_t == "T1b" ~ "T1",
      ajcc_pathologic_t == "T1c" ~ "T1",
      ajcc_pathologic_t == "T2" ~ "T2",
      ajcc_pathologic_t == "T2a" ~ "T2",
      ajcc_pathologic_t == "T2b" ~ "T2",
      ajcc_pathologic_t == "T3" ~ "T3",
      ajcc_pathologic_t == "T3a" ~ "T3",
      ajcc_pathologic_t == "T4" ~ "T4",
      ajcc_pathologic_t == "T4b" ~ "T4",
      ajcc_pathologic_t == "T4d" ~ "T4",
      ajcc_pathologic_t == "TX" ~ "unknown"
    ),
    # Extract just the first two characters (N0, N1, N2, N3, or NX)
    ajcc_pathologic_n = str_extract(ajcc_pathologic_n, "^N[0-3X]"),
    # ajcc_pathologic_m = case_when(
    #   ajcc_pathologic_m == "cM0 (i+)" ~ "M0",
    #   ajcc_pathologic_m == "M0" ~ "M0",
    #   ajcc_pathologic_m == "M1" ~ "M1",
    #   ajcc_pathologic_m == "MX" ~ "unknown"
    # ),
  ) %>% 
  filter(
    treatments_radiation_treatment_type %in% c("radiation_therapy_NOS", "radiation_external_beam", "unknown"),
    ajcc_pathologic_t %in% c("T1", "T2", "T3", "T4"),
    ajcc_pathologic_n %in% c("N0", "N1", "N2", "N3"),
    # ajcc_pathologic_m %in% c("M0", "M1", "unknown")
  ) %>%
  as_tibble()

ggplot()+
  geom_histogram(aes(x = clinical_selected$days_to_last_follow_up / 365), fill = "lightgreen", color = "black")

table(clinical_selected["treatments_pharmaceutical_treatment_or_therapy"])
table(clinical_selected["treatments_radiation_treatment_or_therapy"])
table(clinical_selected["treatments_radiation_treatment_type"])
sum(is.na(clinical_selected["treatments_pharmaceutical_treatment_or_therapy"]))
sum(is.na(clinical_selected["treatments_radiation_treatment_or_therapy"]))
sum(is.na(clinical_selected["treatments_radiation_treatment_type"]))

table(clinical_selected["ajcc_pathologic_t"])
table(clinical_selected["ajcc_pathologic_n"])
# table(clinical_selected["ajcc_pathologic_m"])

# Pulls curated PAM50 subtypes and ER/PR/HER2 status
subtype_brca <- TCGAquery_subtype(tumor = "brca") %>%
  as_tibble()

table(subtype_brca["BRCA_Subtype_PAM50"])

# Register NAs as a separate class and remove the only two observations for which
# BRCA_Subtype_PAM50 is missing
subtype_brca_selected <- subtype_brca %>%
  mutate(
    submitter_id = str_sub(patient, 1, 12),
    BRCA_Pathology = na_if(BRCA_Pathology, "NA"),
    BRCA_Pathology = replace_na(BRCA_Pathology, "Unknown"),
    pathologic_stage = na_if(pathologic_stage, "NA"),
    # pathologic_stage = replace_na(pathologic_stage, "Unknown"),
    BRCA_Subtype_PAM50 = na_if(BRCA_Subtype_PAM50, "NA"),
    BRCA_Subtype_PAM50 = na_if(BRCA_Subtype_PAM50, "Normal"),
    event = ifelse(vital_status == "Dead", 1, 0),
    time = ifelse(event == 1, days_to_death, days_to_last_followup)
  ) %>%
  mutate(
    pathologic_stage = case_when(
      pathologic_stage == "Stage_I" ~ "Stage_I",
      pathologic_stage == "Stage_II" ~ "Stage_II",
      pathologic_stage == "Stage_III" ~ "Stage_III+",
      pathologic_stage == "Stage_IV" ~ "Stage_III+"
    ),
  ) %>%
  filter(!is.na(BRCA_Subtype_PAM50) & !is.na(pathologic_stage)) %>%
  select(submitter_id,
         pathologic_stage, BRCA_Pathology, BRCA_Subtype_PAM50,
         time, event)
table(subtype_brca_selected["BRCA_Subtype_PAM50"])
sum(is.na(subtype_brca_selected["BRCA_Subtype_PAM50"]))

table(subtype_brca_selected["pathologic_stage"])
sum(is.na(subtype_brca_selected["pathologic_stage"]))
table(subtype_brca["pathologic_stage"])

# Merge clinical and subtype data using the patient barcode
clinical_merged <- inner_join(subtype_brca_selected, clinical_selected, by = "submitter_id")

# Ensure the time (in days) is well defined and is positive. Otherwise, discard observations
clinical_merged$time <- as.numeric(clinical_merged$time)
clinical_merged <- clinical_merged[ !is.na(clinical_merged$time) & (clinical_merged$time > 0) , ]

# Split the patients up to this part into a train and a test set
times <- clinical_merged$time
event <- clinical_merged$event

set.seed(1)
# Proportion for training
train_prop <- 0.7
# Indices stratified by delta
train_idx <- unlist(
  lapply(split(seq_along(event), event), function(idx) {
    sample(idx, size = floor(train_prop * length(idx)))
  })
)
# Test indices are the remaining ones
test_idx <- setdiff(seq_along(event), train_idx)

clinical_merged_train <- clinical_merged[ train_idx , ]
clinical_merged_test <- clinical_merged[ test_idx , ]

train_test_equal_classes <- TRUE
for(col in colnames(clinical_merged_train)[-1]){
  print(col)
  if( class( clinical_merged_train[[col]] ) == "character" ){
    cat("Number of unique classes:\n")
    cat(paste0("    Train: ", length(unique(clinical_merged_train[[col]])) , "\n"))
    cat(paste0("    Test: ", length(unique(clinical_merged_test[[col]])) , "\n"))
    if( length(unique(clinical_merged_train[[col]])) != length(unique(clinical_merged_test[[col]])) ){
      train_test_equal_classes <- FALSE
    }
  }
}
if( train_test_equal_classes ){
  cat("Number of classes in train and test are equal!\n")
}else{
  cat("Number of classes in train and test sets differ!\n")
}

# Fetch the curated TCGA-BRCA MAF directly
maf_data <- tcgaLoad(study = "BRCA")
# Extract the mutation count matrix
mut_matrix <- mutCountMatrix(maf = maf_data)
# Extract patients id's from gene mutations table
mut_matrix <- mut_matrix %>%
  t() %>% 
  as.data.frame() %>% 
  rownames_to_column("Tumor_Sample_Barcode") %>%
  mutate(submitter_id = str_sub(Tumor_Sample_Barcode, 1, 12)) %>%
  select(!Tumor_Sample_Barcode)

# Filter only patients that are in the train set
mut_matrix_train <- mut_matrix %>%
  filter( submitter_id %in% clinical_merged_train$submitter_id )
# Filter only patients that are in the test set
mut_matrix_test <- mut_matrix %>%
  filter( submitter_id %in% clinical_merged_test$submitter_id )

# Remove the index column from the train mut_matrix
num_mut_matrix_train <- mut_matrix_train %>% select(!submitter_id)
# Extract the top genes appearing in the train set
top_genes <- names(sort(colSums(num_mut_matrix_train), decreasing = TRUE)[1:50])

# Select the top 50 genes selected in the train set for both train and test sets
mut_matrix_train_filtered <- mut_matrix_train[, c("submitter_id", c(top_genes)) ] %>% 
  as.data.frame()
# Binarize the mutations count matrix to detect 1 or more mutations or none
mut_matrix_train_filtered <- mut_matrix_train_filtered %>%
  mutate(across(all_of(top_genes), ~ ifelse(. > 0, 1, 0)))

mut_matrix_test_filtered <- mut_matrix_test[, c("submitter_id", c(top_genes)) ] %>% 
  as.data.frame()
mut_matrix_test_filtered <- mut_matrix_test_filtered %>%
  mutate(across(all_of(top_genes), ~ ifelse(. > 0, 1, 0)))

# Merge all data into a single table
final_data_train <- inner_join(clinical_merged, mut_matrix_train_filtered, by = "submitter_id")
final_data_test <- inner_join(clinical_merged, mut_matrix_test_filtered, by = "submitter_id")
dim(final_data_train)
dim(final_data_test)

write.csv(final_data_train, file.path(out_dir, "tcga_clinical_and_mutations_train.csv"), row.names = FALSE)
write.csv(final_data_test, file.path(out_dir, "tcga_clinical_and_mutations_test.csv"), row.names = FALSE)
