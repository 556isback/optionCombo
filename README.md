# optionCombo

## description

OptionCombo is a powerful library that enables the discovery of all possible option combinations for a single asset. It takes about 8 mins to compute 57k combinations

## limations

Currently, OptionCombo supports the identification of strategies consisting of two to four option combinations with the same expiry date.

Please note that OptionCombo is designed exclusively for computation purposes. It does not include a method for downloading option data. However, we have included a function in the example file that facilitates downloading and formatting data, serving as a helpful starting point.

## usage

OptionCombo is not yet available on PyPI. However, you can install it using the provided release file.

### 1. Import OptionCombo

To begin, import the necessary modules:

```
from optionCombo import optionModel,preInit
```
To utilize the computation function, you will need option data from your preferred source. You can find a data sample in the test file, as well as an example in the example/example.ipynb file:
```
option_data
pricedata  # price data (not necessary but used for determining the expected range)
```

### 2. Precompute Greeks of Single Option

In this step, you will need to select the expiry date for your option strategy:
```
expirDate = '2023-06-30T00:00:00.000000000' # e.g., choose the desired expiry date
preOption, strikes, price = preInit.Prep( expirDate, optionDf = option_data, priceDf = pricedata )
```

### 3. Compute Greeks and Premium of Option Combinations

Next, input the precomputed results into the model finder. Please note that if the number of combinations exceeds 100,000, it may take a considerable amount of time to compute (approximately 120 iterations per second). The function returns a pandas DataFrame containing all the combinations based on the provided parameters. You can filter the results using the statistics provided in the DataFrame:

```
model = optionModel.option_model(price, joined1,preOption,optiontypes = [[1,1,1]],tradetypes=[[1,-1,1]], maxquantity=1)  # Init
df = model.options_model_finder()
```

### 4. Plotting

Finally, OptionCombo offers a method to visualize the payoff curve of the option combinations. You only need the "para" column from the resulting DataFrame:

```
for para in df.para[:5]:
    model.model_plot(para)
```
For a detailed tutorial, please refer to the example file, which includes functions for collecting and formatting data. It provides a ready-to-use tutorial if you intend to trade on Deribit.

## to verify the result
if you are trading crypto option 
you can use [Greeks.live](https://www.greeks.live/#/deribit/tools/pv/ETH) to verify the result, but its going to require a [Deribit](https://www.deribit.com/?reg=18011.8749&q=home) account to register.
















## disclaimer

This software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

Option trading is risky and may not be suitable for all investors. Users should be aware of the risks involved and should only invest funds they can afford to lose.

The creator of this project is not responsible for any losses that users may encounter while using this software.
