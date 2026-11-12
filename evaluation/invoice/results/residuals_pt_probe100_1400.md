# Real Portuguese probe — residual categories

Documents: 100; with at least one wrong field: 97.

| Category | Count |
|---|---:|
| `total_double_counted_tax` | 50 |
| `vendor_other` | 27 |
| `number_other` | 26 |
| `total_other` | 24 |
| `tax_other` | 19 |
| `number_digit_or_shape` | 17 |
| `vendor_alias_contains` | 12 |
| `number_empty` | 11 |
| `total_dot_decimal_x1000` | 5 |
| `currency_wrong_iso` | 4 |
| `date_wrong` | 4 |
| `tax_missing` | 1 |
| `total_missing` | 1 |
| `total_selected_line_amount` | 1 |
| `vendor_empty` | 1 |

## Examples

### `currency_wrong_iso`
- `ptreal-company-20210322_163546` currency: expected `EUR`, got `BRL`
- `ptreal-company-20210428_204159` currency: expected `EUR`, got `BRL`
- `ptreal-company-20210507_215847` currency: expected `EUR`, got `E`
- `ptreal-personal-20210215_164929` currency: expected `EUR`, got ``

### `date_wrong`
- `ptreal-company-20210428_165519` invoice_date: expected `2019-03-12`, got `2019-03-02`
- `ptreal-company-20210429_181056` invoice_date: expected `2019-06-18`, got `2019-08-18`
- `ptreal-company-20210429_181303` invoice_date: expected `2019-09-24`, got `2010-09-24`
- `ptreal-personal-20210215_164929` invoice_date: expected `2020-11-28`, got `None`

### `number_digit_or_shape`
- `ptreal-company-20210318_111820` invoice_number: expected `A/28564`, got `A/28664`
- `ptreal-company-20210322_163546` invoice_number: expected `3F5C111219/753`, got `F1 3FSC111219753`
- `ptreal-company-20210428_144401` invoice_number: expected `1A1901/1404`, got `FS 1A1901/1404`
- `ptreal-company-20210428_165519` invoice_number: expected `1240722019030000/111392`, got `1240722519030000/111392`
- `ptreal-company-20210428_181827` invoice_number: expected `FS 02111061902/0033496`, got `FS 0211061902/0033496`
- `ptreal-company-20210428_224621` invoice_number: expected `1A1901/4146`, got `FS 1A1901/4146`
- `ptreal-company-20210428_225028` invoice_number: expected `32/55047`, got `32 32/55047`
- `ptreal-company-20210428_232117` invoice_number: expected `004/0007018`, got `N. 004/0007Q18`

### `number_empty`
- `ptreal-company-20210318_172108` invoice_number: expected `00675829`, got ``
- `ptreal-company-20210323_164802` invoice_number: expected `012010004335FAAA00000115032019/000004454`, got ``
- `ptreal-company-20210323_170017` invoice_number: expected `022010004335FAAA00000114422019/000000429`, got ``
- `ptreal-company-20210429_200119` invoice_number: expected `FT 04400021901010617/003644`, got ``
- `ptreal-company-20210429_205320` invoice_number: expected `FT 04420071902222312/000709`, got ``
- `ptreal-company-20210507_214450` invoice_number: expected `012010004110FAAA00000112/22017/000012253`, got ``
- `ptreal-company-20210507_214526` invoice_number: expected `012010003305FAAB00009656272017/000011618`, got ``
- `ptreal-company-20210508_174017` invoice_number: expected `FTR 6/17402`, got ``

### `number_other`
- `ptreal-company-20210318_171853` invoice_number: expected `0086466`, got `501649FF20`
- `ptreal-company-20210318_172727` invoice_number: expected `00513628`, got `501649FF20`
- `ptreal-company-20210318_174038` invoice_number: expected `A0000000032010`, got `LISBOA 2`
- `ptreal-company-20210318_205001` invoice_number: expected `030101732516200003/00005`, got `981`
- `ptreal-company-20210318_205159` invoice_number: expected `00102212`, got `2795-684 CARNAXIDE`
- `ptreal-company-20210322_125040` invoice_number: expected `FP001.0008038`, got `509816975`
- `ptreal-company-20210322_163939` invoice_number: expected `FT 2545/6423`, got `2545-314659`
- `ptreal-company-20210323_171049` invoice_number: expected `FT 1A1905E1/33832`, got `FATURA-RECIBO N: FT 1A1905E1/33832`

### `tax_missing`
- `ptreal-personal-20210215_164929` tax: expected `14.35`, got `None`

### `tax_other`
- `ptreal-company-20210322_125040` tax: expected `1.27`, got `1.99`
- `ptreal-company-20210323_194346` tax: expected `0.58`, got `1.2`
- `ptreal-company-20210428_140310` tax: expected `1.46`, got `2.2`
- `ptreal-company-20210428_141142` tax: expected `0.65`, got `3.02`
- `ptreal-company-20210428_223139` tax: expected `3.14`, got `16.8`
- `ptreal-company-20210429_181056` tax: expected `7.1`, got `37.99`
- `ptreal-company-20210429_194828` tax: expected `0.43`, got `2.3`
- `ptreal-company-20210429_204946` tax: expected `0.95`, got `17.0`

### `total_dot_decimal_x1000`
- `ptreal-company-20210318_171853` total: expected `8.88`, got `8880.0`
- `ptreal-company-20210318_172108` total: expected `16.0`, got `16000.0`
- `ptreal-company-20210318_172727` total: expected `119.22`, got `119220.0`
- `ptreal-company-20210318_174038` total: expected `9.25`, got `9250.0`
- `ptreal-company-20210318_205159` total: expected `21.67`, got `21670.0`

### `total_double_counted_tax`
- `ptreal-company-20210318_111820` total: expected `32.0`, got `37.98`
- `ptreal-company-20210318_205001` total: expected `47.96`, got `56.93`
- `ptreal-company-20210322_172402` total: expected `20.0`, got `21.13`
- `ptreal-company-20210323_170017` total: expected `60.0`, got `71.22`
- `ptreal-company-20210323_170843` total: expected `14.15`, got `16.8`
- `ptreal-company-20210323_171049` total: expected `6.0`, got `7.12`
- `ptreal-company-20210323_185919` total: expected `29.55`, got `35.08`
- `ptreal-company-20210323_190834` total: expected `1.6`, got `1.78`

### `total_missing`
- `ptreal-personal-20210215_164929` total: expected `76.72`, got `None`

### `total_other`
- `ptreal-company-20210322_125040` total: expected `6.77`, got `10.75`
- `ptreal-company-20210322_135306` total: expected `20.95`, got `4190.0`
- `ptreal-company-20210323_175524` total: expected `11.1`, got `15.26`
- `ptreal-company-20210323_183059` total: expected `0.7`, got `128.7`
- `ptreal-company-20210428_140310` total: expected `7.8`, got `12.2`
- `ptreal-company-20210428_141142` total: expected `3.49`, got `10.0`
- `ptreal-company-20210428_145526` total: expected `2.7`, got `10.8`
- `ptreal-company-20210428_151555` total: expected `29.4`, got `64.3`

### `total_selected_line_amount`
- `ptreal-company-20210428_144615` total: expected `5.2`, got `520.0`

### `vendor_alias_contains`
- `ptreal-company-20210323_185919` vendor_name: expected `drogaria semino`, got `DROGARIA SEMINO SOC. FERRAMENTAS & FERRAGENS SEMINO LDA`
- `ptreal-company-20210428_141142` vendor_name: expected `jumbo`, got `JUMBO AMOREIRAS`
- `ptreal-company-20210428_183109` vendor_name: expected `lelimel`, got `LELIMEL-MATERIAIS DE CONSTRUÇÃO,LDA`
- `ptreal-company-20210429_181056` vendor_name: expected `bp`, got `BP MATRA NORTH`
- `ptreal-company-20210507_200800` vendor_name: expected `lenave`, got `Lenave Comércio Industrial e Representações Ltda.`
- `ptreal-company-20210507_212923` vendor_name: expected `siletrica`, got `SILETRICA-Soc Electr Portuguesa Lda`
- `ptreal-company-20210508_205115` vendor_name: expected `el corte ingles`, got `EL CORTE INGLES - GRANDES ARMAZENS S A`
- `ptreal-company-20210508_211208` vendor_name: expected `aki`, got `AKI Mafra`

### `vendor_empty`
- `ptreal-personal-20210215_164929` vendor_name: expected `zara home`, got `None`

### `vendor_other`
- `ptreal-company-20210318_171853` vendor_name: expected `siletrica lda`, got `MILLENNIUM BCP`
- `ptreal-company-20210318_172108` vendor_name: expected `penha da graca ferragens unip, lda`, got `PENHA GRACA`
- `ptreal-company-20210322_163546` vendor_name: expected `intermarche`, got `DISTRIBUIDORA SUPERMERCADOS LDA`
- `ptreal-company-20210322_163939` vendor_name: expected `cp comboios de portugal, e. p. e.`, got `FlikeHomeAlgarve,Lda`
- `ptreal-company-20210323_164802` vendor_name: expected `galp`, got `A.S. LOULE (FARO/LOULE) - 11503`
- `ptreal-company-20210323_170017` vendor_name: expected `galp`, got `A.S. LAGOS (FARG/SAGRES) - 11442`
- `ptreal-company-20210323_173000` vendor_name: expected `carlos eduardo pires pereira`, got `LIKEHOME ALGARVE, LDA`
- `ptreal-company-20210323_175524` vendor_name: expected `shopping loule`, got `LEIA CREIA LDA`
