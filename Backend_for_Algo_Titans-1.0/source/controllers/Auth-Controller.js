const bcrypt = require('bcrypt')
const jwt = require('jsonwebtoken')
const UserModel = require("../models/User-model")
const { GenerateToken } = require('../utils/Generate-Token')

module.exports.CreateAccount = async (req, res) => {

    try {
        let { FullName, Email, Password } = req.body

        let user = await UserModel.findOne({ Email: Email })
        if (user) return res.redirect("/user_exists")


        bcrypt.genSalt(10, (err, salt) => {
            bcrypt.hash(Password, salt, async function (err, hash) {
                if (err) return (err.message)

                else {
                    let user = await UserModel.create({
                        FullName,
                        Email,
                        Password: hash
                    })
                    let token = GenerateToken(user)
                    res.cookie("token", token)

                    res.send(user)
                    // res.redirect("/home")
                }
            });
        });


    }
    catch (err) {
        res.send(err.message);
    }

}

module.exports.Login = async (req, res) => {

    let { Email, Password } = req.body

    let user = await UserModel.findOne({ Email: Email })

    if (!user) return res.redirect("/something_went_wrong")
    // if (!user) return res.send("User does not exist!!!")

    bcrypt.compare(Password, user.Password, (err, result) => {
        if (result === true) {
            let token = GenerateToken(user)
            res.cookie("token", token)
            // res.redirect("/home")
            res.send("loggedIn")
        }
        else {
            res.redirect("/something_went_wrong")
        }
    });

} 