# Set working directory
setwd("~/codes/metabric_aft_models/data")

library(cBioPortalData)
library(tidyverse)
library(MultiAssayExperiment)
library(survival)

options(timeout = max(600, getOption("timeout")))
# download.file("https://datahub.assets.cbioportal.org/brca_metabric.tar.gz", "metabric.tar.gz")

# Extract patient mutations data
untar("metabric.tar.gz", files = "brca_metabric/data_mutations.txt")
# Read data mutations file
metabric_raw_maf <- read_tsv("brca_metabric/data_mutations.txt", comment = "#")

# Build gene mutations count matrix
mutation_count_matrix <- metabric_raw_maf %>%
  select(Hugo_Symbol, Tumor_Sample_Barcode, Variant_Classification) %>%
  filter(Variant_Classification != "Silent") %>%
  group_by(Tumor_Sample_Barcode, Hugo_Symbol) %>%
  summarise(mut_count = n(), .groups = "drop") %>%
  pivot_wider(
    names_from = Hugo_Symbol,
    values_from = mut_count,
    values_fill = list(mut_count = 0)
  )

# Extract patient clinical data
untar("metabric.tar.gz", files = "brca_metabric/data_clinical_patient.txt")
# Read clinical data
metabric_clinical_raw <- read_tsv("brca_metabric/data_clinical_patient.txt", comment = "#")

# Clean up the clinical data and select the crucial variables
metabric_clinical_clean <- metabric_clinical_raw %>%
  select(
    PATIENT_ID,
    AGE_AT_DIAGNOSIS,
    RFS_MONTHS,
    RFS_STATUS,
    CLAUDIN_SUBTYPE, # PAM50 subtype
    AGE_AT_DIAGNOSIS,
    NPI
  ) %>%
  mutate(
    time_months = as.numeric(RFS_MONTHS),
    delta = case_when(
      RFS_STATUS == "1:Recurred" ~ 1,
      RFS_STATUS == "0:Not Recurred" ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(time_months), time_months > 0)

# Merge the clinical data with your mutation counts
final_dataset <- metabric_clinical_clean %>%
  inner_join(mutation_count_matrix, by = c("PATIENT_ID" = "Tumor_Sample_Barcode"))

# View the final pristine dataset
View(final_dataset)

# Calculate the median using the reversed delta
reverse_km <- survfit(Surv(as.numeric(time_months), 1-delta) ~ 1, data = metabric_clinical_clean)
print(reverse_km)

# We see a median follow-up time of about 158 months, or 13 years and 2 months
write.csv(final_dataset, "00_clean/metabric_full_data.csv", row.names = FALSE)
