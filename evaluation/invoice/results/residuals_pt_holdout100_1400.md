# Real Portuguese probe — residual categories

Documents: 100; with at least one wrong field: 78.

| Category | Count |
|---|---:|
| `tax_other` | 30 |
| `number_other` | 27 |
| `vendor_other` | 25 |
| `number_empty` | 19 |
| `vendor_alias_contains` | 13 |
| `number_digit_or_shape` | 11 |
| `total_double_counted_tax` | 11 |
| `date_wrong` | 7 |
| `total_other` | 6 |
| `currency_wrong_iso` | 4 |
| `total_missing` | 3 |
| `total_selected_line_amount` | 3 |
| `tax_missing` | 2 |
| `total_dot_decimal_x1000` | 1 |
| `vendor_empty` | 1 |

## Examples

### `currency_wrong_iso`
- `ptreal-company-20210323_184527` currency: expected `EUR`, got ``
- `ptreal-company-20210507_203411` currency: expected `EUR`, got `BRL`
- `ptreal-personal-20210215_165018` currency: expected `EUR`, got ``
- `ptreal-personal-20210509_002620` currency: expected `EUR`, got `BRL`

### `date_wrong`
- `ptreal-company-20210323_184527` invoice_date: expected `2019-03-07`, got `None`
- `ptreal-company-20210428_005103` invoice_date: expected `2019-02-10`, got `None`
- `ptreal-company-20210428_142429` invoice_date: expected `2019-12-13`, got `2019-02-19`
- `ptreal-company-20210428_195539` invoice_date: expected `2019-01-12`, got `2019-01-19`
- `ptreal-company-20210508_144523` invoice_date: expected `2018-10-06`, got `None`
- `ptreal-company-20210508_173053` invoice_date: expected `2018-11-15`, got `None`
- `ptreal-personal-20210215_165018` invoice_date: expected `2020-11-25`, got `None`

### `number_digit_or_shape`
- `ptreal-company-20210320_171557` invoice_number: expected `1A1901/44`, got `FT 1A1901/44`
- `ptreal-company-20210323_165310` invoice_number: expected `FT 0E06011219/4248`, got `FT 006011219/4248`
- `ptreal-company-20210323_172831` invoice_number: expected `VD002.001994`, got `VD002.0019944`
- `ptreal-company-20210323_175637` invoice_number: expected `45300520190053/122`, got `FAC 45300520190053/122`
- `ptreal-company-20210428_224937` invoice_number: expected `004/0007113`, got `N. 004/0007113`
- `ptreal-company-20210508_173905` invoice_number: expected `FS HRX003/009275`, got `FT HRX003/009275`
- `ptreal-company-20210508_202014` invoice_number: expected `PO 001/1806719`, got `Nº:PO 001/1806719`
- `ptreal-personal-20210215_163617` invoice_number: expected `F 0009/27171`, got `F U009/27171`

### `number_empty`
- `ptreal-company-20210322_164701` invoice_number: expected `012010004335FAAA00000114422019/000001545`, got ``
- `ptreal-company-20210323_183923` invoice_number: expected `FS 03650551901010620/016314`, got ``
- `ptreal-company-20210428_005103` invoice_number: expected `FT 04510051901010617/000417`, got ``
- `ptreal-company-20210428_194905` invoice_number: expected `FS 04420051901010621/046756`, got ``
- `ptreal-company-20210428_200430` invoice_number: expected `012010004110FAAA00000112722019/000005113`, got ``
- `ptreal-company-20210429_200454` invoice_number: expected `FT 0440002161010517/003865`, got ``
- `ptreal-company-20210429_200637` invoice_number: expected `FT 04400021901010617/003865`, got ``
- `ptreal-company-20210429_201903` invoice_number: expected `FT 04400011901010617/003334`, got ``

### `number_other`
- `ptreal-company-20210318_205127` invoice_number: expected `00102213`, got `2795-684 CARNAXIDE`
- `ptreal-company-20210322_172436` invoice_number: expected `0EAA061219/134`, got `91710612836`
- `ptreal-company-20210323_174503` invoice_number: expected `VD002.0011399`, got `9003.0518`
- `ptreal-company-20210323_184527` invoice_number: expected `FS 03650081901010623/000234`, got `0000000000000000000000000000000`
- `ptreal-company-20210428_135731` invoice_number: expected `FR 270652001/042320`, got `POS:2 Ref:483 Op:Jandira Vaz`
- `ptreal-company-20210428_142429` invoice_number: expected `FS HRX003/036250`, got `P1503630330`
- `ptreal-company-20210428_194558` invoice_number: expected `FT 04420051901010621/001251`, got `262530`
- `ptreal-company-20210428_202721` invoice_number: expected `FT 2019A17/3735`, got `3735 - 17`

### `tax_missing`
- `ptreal-company-20210323_184527` tax: expected `0.67`, got `None`
- `ptreal-personal-20210215_165018` tax: expected `2.42`, got `None`

### `tax_other`
- `ptreal-company-20210320_164056` tax: expected `14.37`, got `1609.0`
- `ptreal-company-20210321_120551` tax: expected `14.37`, got `1609.0`
- `ptreal-company-20210322_172436` tax: expected `4.67`, got `20.32`
- `ptreal-company-20210323_175637` tax: expected `0.9`, got `3.9`
- `ptreal-company-20210326_113812` tax: expected `1.78`, got `9.51`
- `ptreal-company-20210428_005103` tax: expected `0.43`, got `3.74`
- `ptreal-company-20210428_135731` tax: expected `1.87`, got `23.0`
- `ptreal-company-20210428_141337` tax: expected `3.72`, got `0.1`

### `total_dot_decimal_x1000`
- `ptreal-personal-20210509_000204` total: expected `43.2`, got `43200.0`

### `total_double_counted_tax`
- `ptreal-company-20210323_172831` total: expected `15.1`, got `17.92`
- `ptreal-company-20210323_174503` total: expected `46.6`, got `55.31`
- `ptreal-company-20210323_175637` total: expected `4.8`, got `8.7`
- `ptreal-company-20210323_183923` total: expected `3.88`, got `4.19`
- `ptreal-company-20210428_141337` total: expected `19.9`, got `20.0`
- `ptreal-company-20210428_200430` total: expected `10.0`, got `11.87`
- `ptreal-company-20210428_235141` total: expected `155.0`, got `180.98`
- `ptreal-company-20210429_203900` total: expected `5.58`, got `6.63`

### `total_missing`
- `ptreal-company-20210323_184527` total: expected `29.04`, got `None`
- `ptreal-company-20210429_192628` total: expected `2.9`, got `None`
- `ptreal-personal-20210215_165018` total: expected `12.95`, got `None`

### `total_other`
- `ptreal-company-20210322_164701` total: expected `20.0`, got `23.74`
- `ptreal-company-20210428_142429` total: expected `29.99`, got `79.99`
- `ptreal-company-20210508_204423` total: expected `25.01`, got `29.67`
- `ptreal-personal-20210215_164339` total: expected `2.83`, got `7.17`
- `ptreal-personal-20210215_164422` total: expected `30.81`, got `50.86`
- `ptreal-personal-20210509_002620` total: expected `18.9`, got `1890.0`

### `total_selected_line_amount`
- `ptreal-company-20210318_205127` total: expected `293.13`, got `29313.0`
- `ptreal-company-20210323_165310` total: expected `9.99`, got `23.0`
- `ptreal-company-20210508_144523` total: expected `1.41`, got `20.0`

### `vendor_alias_contains`
- `ptreal-company-20210320_164056` vendor_name: expected `bp`, got `BP DUARTE PACHECO`
- `ptreal-company-20210322_122759` vendor_name: expected `ctt expresso`, got `CTT Expresso - Serviços Postais e Logistica, S.A.`
- `ptreal-company-20210323_194951` vendor_name: expected `aki`, got `AKI Portimao`
- `ptreal-company-20210428_135731` vendor_name: expected `bp`, got `BP DOMINGOS SEQUEIRA`
- `ptreal-company-20210428_235625` vendor_name: expected `bp`, got `BP PADRE CRUZ`
- `ptreal-company-20210429_200028` vendor_name: expected `bp`, got `BP DOMINGOS SEQUIRA`
- `ptreal-company-20210429_205932` vendor_name: expected `h3`, got `H3 Portugal, LDA.`
- `ptreal-company-20210507_202531` vendor_name: expected `worten`, got `WORTEN LOULE`

### `vendor_empty`
- `ptreal-personal-20210215_165018` vendor_name: expected `zara`, got `None`

### `vendor_other`
- `ptreal-company-20210322_164701` vendor_name: expected `galp`, got `A.S. LAGOS (FARO/SAGRES)`
- `ptreal-company-20210323_165310` vendor_name: expected `bricomarché`, got `BRICO LAGOS`
- `ptreal-company-20210323_172831` vendor_name: expected `f. raminhos & filhos, lda`, got `FLIKEHOME ALGARVE, LDA`
- `ptreal-company-20210326_113812` vendor_name: expected `ferragens g lemos lda`, got `Flikehome Mediacao Imobilia`
- `ptreal-company-20210428_142929` vendor_name: expected `zhang huanzhen`, got `FLIKE HOME`
- `ptreal-company-20210428_194558` vendor_name: expected `pingo doce`, got `FRANCISCO METRASS`
- `ptreal-company-20210429_194705` vendor_name: expected `ferragens g lemos lda`, got `Flikhome - Mediacao Imobiliar`
- `ptreal-company-20210429_200454` vendor_name: expected `pingo doce`, got `TOMAS RIBEIRO`
