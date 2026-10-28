
library(survival)
library(flexsurv)
library(ggplot2)

setwd("~/codes/metabric_aft_models/notebooks")

# ------------------------------------
# ----------- Loading data -----------
# ------------------------------------
df_train <- read.csv("../data/00_clean/metabric_train_data.csv")
df_test <- read.csv("../data/00_clean/metabric_test_data.csv")
df_train["time"] <- df_train["time"] / 12
df_test["time"] <- df_test["time"] / 12

# --------------------------------------------------
# ----------- Defining Models Structures -----------
# --------------------------------------------------

pllogis_cure <- function(q, mu, sigma, p, lower.tail = TRUE, log.p = FALSE) {
  z <- (log(q) - mu) / sigma
  
  # Sobrevivência da base Log-Logística: 1 / (1 + exp(z))
  S1 <- 1 / (1 + exp(z))
  S_pop <- p + (1 - p) * S1
  
  res <- if (lower.tail) 1 - S_pop else S_pop
  if (log.p) return(log(res))
  return(res)
}

dllogis_cure <- function(x, mu, sigma, p, log = FALSE) {
  z <- (log(x) - mu) / sigma
  
  # Log-Densidade usando log1p(exp(z)) para máxima estabilidade (equivale ao softplus)
  log_f1 <- -log(x) - log(sigma) + z - 2 * log1p(exp(z))
  
  if (log) {
    # Retorna a log-verossimilhança diretamente
    return(log(1 - p) + log_f1)
  } else {
    # Retorna a densidade na escala original
    return((1 - p) * exp(log_f1))
  }
}

custom_ll_cure <- list(
  name = "llogis_cure",
  pars = c("mu", "sigma", "p"),
  location = "mu",
  transforms = c(identity, log, qlogis), # mu livre, sigma > 0 (log link), p in (0,1) (logit link)
  inv.transforms = c(identity, exp, plogis),
  inits = function(t) c(mean(log(t)), 1, 0.2) # Chute inicial: p = 20% de cura
)

fit_ll <- flexsurvreg(
  Surv(time, delta) ~ age_at_diagnosis + pam50_subtype + npi, 
  anc = list(
    sigma = ~ pam50_subtype,                                   
    p = ~ age_at_diagnosis + pam50_subtype + npi               
  ),
  data = df_train,
  dist = custom_ll_cure
)

S_ll <- summary(fit_ll)
plot(fit_ll, main="Log-Logistic Cure vs Kaplan-Meier", xlab="Anos", ylab="Sobrevivência")


# --------------------------------------
# ----------- Building Plots -----------
# --------------------------------------

km_fit <- survfit(Surv(time, delta) ~ pam50_subtype, data = df_train)

# Extraindo os dados do objeto survfit para um data.frame amigável ao ggplot2
df_km <- data.frame(
  time = km_fit$time,
  surv = km_fit$surv,
  strata = rep(names(km_fit$strata), km_fit$strata)
)

# Limpar o nome da classe (remove o prefixo 'pam50_subtype=')
df_km$pam50_subtype <- gsub("pam50_subtype=", "", df_km$strata)

# Para geom_step desenhar corretamente desde o tempo 0 (sobrevivência = 1)
df_km_start <- data.frame(
  time = 0,
  surv = 1,
  strata = unique(df_km$strata),
  pam50_subtype = unique(df_km$pam50_subtype)
)
df_km <- rbind(df_km_start, df_km)
df_km <- df_km[order(df_km$pam50_subtype, df_km$time), ]

# =================================================================
# 2. DADOS DO MODELO AFT COM CURA (Predições)
# =================================================================
# Obtendo a média das variáveis contínuas para padronizar as curvas
mean_age <- mean(df_train$age_at_diagnosis, na.rm = TRUE)
mean_npi <- mean(df_train$npi, na.rm = TRUE)
pam50_levels <- unique(df_train$pam50_subtype)

# Criando um 'newdata' com um paciente médio para cada subtipo PAM50
newdata_pred <- data.frame(
  pam50_subtype = pam50_levels,
  age_at_diagnosis = mean_age,
  npi = mean_npi
)

# Vetor de tempo: do tempo 0 até o maior tempo de observação, 200 pontos garantem suavidade
t_grid <- seq(0, max(df_train$time, na.rm = TRUE), length.out = 200)

# flexsurv gera as predições de forma limpa usando a função summary
pred_list <- summary(fit_ll, newdata = newdata_pred, t = t_grid)

# Juntando as listas do summary em um único data.frame
df_pred <- data.frame()
for (i in seq_along(pred_list)) {
  temp_df <- pred_list[[i]]
  # A ordem no summary bate com a ordem das linhas do newdata
  temp_df$pam50_subtype <- newdata_pred$pam50_subtype[i] 
  df_pred <- rbind(df_pred, temp_df)
}


# =================================================================
# 3. PLOTAGEM COM GGPLOT2
# =================================================================
p <- ggplot() +
  # Linhas Empíricas (Kaplan-Meier) - Degraus semi-transparentes e contínuos
  geom_step(data = df_km, aes(x = time, y = surv, color = pam50_subtype),
            linewidth = 0.9, alpha = 0.5, linetype = "solid") +
  
  # Linhas do Modelo (Weibull Cure) - Suaves e tracejadas
  geom_line(data = df_pred, aes(x = time, y = est, color = pam50_subtype),
            linewidth = 1.2, linetype = "dashed") +
  
  # Escalas e Estética
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(breaks = seq(0, max(df_train$time, na.rm=TRUE), by = 5)) +
  
  # Configuração de temas e textos
  labs(
    title = "Sobrevivência Câncer de Mama: Modelo Weibull Cure vs Kaplan-Meier",
    subtitle = "Linha Contínua: Kaplan-Meier | Tracejada: Modelo AFT (Idade e NPI fixados na média)",
    x = "Tempo (Anos após diagnóstico)",
    y = "Probabilidade de Sobrevivência",
    color = "Subtipo PAM50"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    legend.background = element_rect(fill = "white", color = NA),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold")
  )

# Renderizar o gráfico
print(p)

# Instale o pacote knitr se não o tiver: install.packages("knitr")
library(knitr)

# 1. Extrair a matriz de resultados do objeto flexsurvreg
coef_matrix <- fit_ll$res

# 2. Converter para data.frame e mover os nomes das linhas (parâmetros) para uma coluna
df_coef <- as.data.frame(coef_matrix)
df_coef$Termo <- rownames(df_coef)

# 3. Calcular a Estatística Z e o Valor-p (Teste de Wald)
df_coef$Z_value <- df_coef$est / df_coef$se
df_coef$P_value <- 2 * pnorm(abs(df_coef$Z_value), lower.tail = FALSE)

# 4. Limpar e reorganizar as colunas
df_coef <- df_coef[, c("Termo", "est", "se", "L95%", "U95%", "Z_value", "P_value")]
collames(df_coef) <- c("Covariável", "Estimativa (Beta)", "Erro Padrão", 
                       "IC 95% Inf", "IC 95% Sup", "Valor Z", "Valor-p")

# 5. Criar uma coluna com Valor-p formatado para tabelas acadêmicas
df_coef$`Valor-p` <- ifelse(
  df_coef$`Valor-p` < 0.001, 
  "< 0.001", 
  sprintf("%.4f", df_coef$`Valor-p`)
)

# 6. Remover os nomes das linhas originais para o print ficar limpo
rownames(df_coef) <- NULL

# 7. Imprimir a tabela formatada no console (ou Markdown/LaTeX)
kable(df_coef, digits = 4, align = "lrrrrrr", 
      caption = "Coeficientes Estimados - Modelo AFT Weibull com Fração de Cura")




